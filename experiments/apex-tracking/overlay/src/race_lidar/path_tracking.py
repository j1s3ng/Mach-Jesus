"""Opt-in selected-path tracking and scheduled-speed control.

All geometry uses the supplied original progress and map, plus measured ego
pose. Passing approval remains the caller's measured-traffic guard decision.
"""
from __future__ import annotations

import math
from collections import deque
import numpy as np
from race_lidar.hotlap_driver import PurePursuit, longitudinal_control, wrap_angle
from race_lidar.continuous_geometry import ContinuousGeometry
from race_lidar.rejoin_trajectory import build_rejoin, validate_rejoin


def constant_offset_route_status(reference, progress, offset, preview_m=20., minimum_scale=.20):
    """Reject a fixed Frenet lane that collapses or reverses around an apex."""
    preview_m = max(0.,float(preview_m))
    locations = np.linspace(float(progress),float(progress)+preview_m,
        max(2,int(math.ceil(preview_m/2.))+1))
    scale = np.array([1.-float(offset)*reference.sample(s)[3] for s in locations])
    index = int(np.argmin(scale))
    return {'trackable':bool(np.isfinite(scale).all() and scale[index]>minimum_scale),
        'minimum_parallel_scale':float(scale[index]),'minimum_scale_progress_m':float(locations[index]),
        'preview_m':preview_m,'minimum_allowed_scale':float(minimum_scale)}


class PathTrackingController:
    VERSION = 'selected-path-feedback-v1'

    def __init__(self, config=None, *, wheelbase=3.2):
        self.config = dict(config or {})
        self.wheelbase = float(wheelbase)
        self.max_steer = float(self.config.get('max_steer_rad', .5))
        self.steer_rate = float(self.config.get('steer_rate_radps', 1.))
        self.heading_gain = float(self.config.get('heading_gain', .9))
        self.lateral_gain = float(self.config.get('lateral_gain', 1.8))
        self.speed_softening = float(self.config.get('speed_softening_mps', 4.))
        self.nominal_capture = float(self.config.get('nominal_capture_m', 1.25))
        self.guard_tolerance = float(self.config.get('guard_approval_tolerance_m', .12))
        self.preview_m = float(self.config.get('profile_preview_m', 0.))
        self.steering_method = str(self.config.get('steering_method','curvature_feedback'))
        self.kinematic_steering_limit_enabled = bool(self.config.get('kinematic_steering_limit_enabled',True))
        self.understeer_gradient = float(self.config.get('understeer_gradient_s2_per_m',0.))
        self.curvature_smoothing_sigma = float(self.config.get('feedforward_curvature_smoothing_sigma_m',0.))
        self.continuous_reference_sigma = float(self.config.get(
            'continuous_reference_feedforward_sigma_m',self.curvature_smoothing_sigma))
        self.coasting_brake_floor = float(self.config.get('rejoin_trajectory',{}).get(
            'coasting_brake_floor_mps2',0.))
        if (not math.isfinite(self.continuous_reference_sigma) or not 0.<=self.continuous_reference_sigma<=4.
                or not math.isfinite(self.coasting_brake_floor) or not 0.<=self.coasting_brake_floor<=8.):
            raise ValueError('Continuous reference sigma must be finite in[0,4]m and coasting brake floor in[0,8]m/s2')
        self.continuous_geometry_enabled = bool(self.config.get('continuous_reference_geometry_enabled',False))
        self._continuous_geometry_cache = {}
        self.rejoin_configuration = dict(self.config.get('rejoin_trajectory',{}))
        self.rejoin_enabled = bool(self.rejoin_configuration.get('enabled',False))
        if self.rejoin_enabled and not self.continuous_geometry_enabled:
            raise ValueError('Rejoin requires continuous reference geometry')
        self._active_rejoin = None
        self._rejoin_context_approved = False
        self._rejoin_context_longitudinal_constraint = None
        self._coasting_measurement_time = None
        self._rejoin_plan_serial = 0
        self._active_rejoin_provenance = None
        self._rejoin_clock = 0.
        self._last_rejoin_attempt_clock = -math.inf
        self._rejoin_last_report = {}
        if not math.isfinite(self.curvature_smoothing_sigma) or not 0.<=self.curvature_smoothing_sigma<=4.:
            raise ValueError('feedforward_curvature_smoothing_sigma_m must be finite within[0,4]')
        # These cache map-derived quadrature only; there is no temporal filter
        # state in the feedback, avoidance target, or actuator command.
        self._curvature_table_key = None
        self._curvature_table = None
        self.longitudinal_friction_estimator_enabled = bool(self.config.get('longitudinal_friction_estimator_enabled',False))
        self.longitudinal_accel_filter_tau = float(self.config.get('longitudinal_accel_filter_tau_s',.10))
        median_samples=self.config.get('longitudinal_accel_median_samples',5)
        if (type(median_samples) is not int or median_samples not in (3,5,7,9)
                or not math.isfinite(self.longitudinal_accel_filter_tau)
                or not .02<=self.longitudinal_accel_filter_tau<=.5):
            raise ValueError('Longitudinal estimator requires odd median_samples3..9 and finite tau within[.02,.5]s')
        if self.longitudinal_friction_estimator_enabled and not self.kinematic_steering_limit_enabled:
            raise ValueError('Longitudinal friction estimator requires the final steering budget enabled')
        self._longitudinal_accel_samples=deque(maxlen=median_samples)
        self._filtered_longitudinal_accel=None
        self.gross_heading_hold_enabled=bool(self.config.get('gross_heading_hold_enabled',False))
        if not math.isfinite(self.understeer_gradient) or not 0.<=self.understeer_gradient<=.005:
            raise ValueError('understeer_gradient_s2_per_m must be finite within[0,.005]')
        if self.steering_method not in {'curvature_feedback','pure_pursuit'}:
            raise ValueError('steering_method must be curvature_feedback or pure_pursuit')
        self.yaw_stability_configuration = dict(self.config.get('yaw_sideslip_stability',{}))
        self.yaw_stability_enabled = bool(self.yaw_stability_configuration.get('enabled',False))
        self._yaw_previous_sideslip = None
        self._yaw_step_previous_steer = None
        if self.yaw_stability_enabled:
            if not (self.longitudinal_friction_estimator_enabled and self.kinematic_steering_limit_enabled):
                raise ValueError('Yaw stability requires existing longitudinal and final steering budgets')
            for key,default,low,high in (
                    ('minimum_speed_mps',20.,5.,50.),('minimum_brake_fraction',.05,0.,1.),
                    ('yaw_rate_deadband_rps',.05,0.,.5),('sideslip_deadband_rad',.005,0.,.2),
                    ('sideslip_growth_deadband_rps',.02,0.,1.),('yaw_rate_gain',1.,0.,2.),
                    ('sideslip_gain',.5,0.,1.),('maximum_correction_rad',.08,.001,.2)):
                value=float(self.yaw_stability_configuration.get(key,default))
                if not math.isfinite(value) or not low<=value<=high:
                    raise ValueError('Invalid yaw stability parameter: '+key)
                self.yaw_stability_configuration[key]=value
        self.previous_steer = 0.
        self.previous_lane = None
        self.previous_speed = None
        self.fallback = PurePursuit(wheelbase=wheelbase, max_steer=self.max_steer,
            steer_rate=self.steer_rate)
        values = [self.wheelbase,self.max_steer,self.steer_rate,self.heading_gain,
            self.lateral_gain,self.speed_softening,self.nominal_capture,self.guard_tolerance]
        if not all(math.isfinite(x) and x > 0 for x in values) or not math.isfinite(self.preview_m) or self.preview_m < 0:
            raise ValueError('Tracking parameters must be positive and preview finite/nonnegative')

    def geometry_view(self, source):
        if not self.continuous_geometry_enabled or source is None:
            return source
        key = id(source)
        if key not in self._continuous_geometry_cache:
            self._continuous_geometry_cache[key] = ContinuousGeometry(source)
        return self._continuous_geometry_cache[key]

    def _cancel_rejoin(self, reason):
        self._active_rejoin = None
        self._rejoin_context_approved = False
        self._rejoin_context_longitudinal_constraint = None
        self._coasting_measurement_time = None
        self._active_rejoin_provenance = None
        self._rejoin_last_report = {'accepted':False,'reason':reason}

    def approved_rejoin_cap(self, progress, preview, guard_flags=()):
        if not self.rejoin_enabled or self._active_rejoin is None or not self._rejoin_context_approved:
            return None
        if any(flag in guard_flags for flag in ('predicted_lane_conflict','alongside_lane_occupied','mapped_nominal_route_conflict')):
            self._cancel_rejoin('final_guard_revoked_rejoin')
            return None
        if float(progress) > self._active_rejoin.end_progress_m:
            self._cancel_rejoin('rejoin_completed_return_to_original_nominal')
            return None
        return float(self._active_rejoin.minimum_speed(progress,preview))

    def _rejoin_route_context(self, reference, line, progress, position, yaw, speed,
                              raw_action, observation, road_rule):
        self._rejoin_context_approved = False
        self._rejoin_context_longitudinal_constraint = None
        replacement_predecessor = None
        if line is None or not self.is_nominal(raw_action) or road_rule is None:
            self._cancel_rejoin('nominal_request_or_actual_road_not_available')
            return None
        if self._active_rejoin is not None and progress > self._active_rejoin.end_progress_m:
            self._cancel_rejoin('rejoin_completed_return_to_original_nominal')
        if self._active_rejoin is not None:
            cached_coast_report=None
            report=validate_rejoin(self._active_rejoin,current_progress_m=progress,
                body_speed_mps=speed,observation=observation,road_rule=road_rule,
                ego_xy=np.asarray(position)[:2],ego_yaw=yaw,configuration=self.rejoin_configuration)
            if (not report.get('clear',False)
                    and report.get('reason')=='measured_rejoin_traffic_conflict'
                    and (self.rejoin_configuration.get('allow_coasting_retention',False)
                         or self.rejoin_configuration.get('replan_after_traffic_conflict',False))):
                constrained_configuration=dict(self.rejoin_configuration,
                    longitudinal_constraint='braking_or_coasting')
                constrained=validate_rejoin(self._active_rejoin,current_progress_m=progress,
                    body_speed_mps=speed,observation=observation,road_rule=road_rule,
                    ego_xy=np.asarray(position)[:2],ego_yaw=yaw,
                    configuration=constrained_configuration)
                cached_coast_report=constrained
                if (constrained.get('clear',False)
                        and self.rejoin_configuration.get('allow_coasting_retention',False)):
                    # The caller MUST enforce the reported zero-positive-drive
                    # constraint before actuation. All original road, pose,
                    # speed-envelope and hazard checks remain in the backend.
                    report=constrained
                    self._rejoin_context_longitudinal_constraint='braking_or_coasting'
                    self._coasting_measurement_time=float(observation.get(
                        '_measurement_time_s',self._rejoin_clock))
            if report.get('clear',False) and self._coasting_measurement_time is not None:
                measurement_time=float(observation.get('_measurement_time_s',self._rejoin_clock))
                if measurement_time <= self._coasting_measurement_time:
                    # A full acceleration tube also proves its coasting subset.
                    # Keep the restrictive actuation contract until fresh data
                    # and another full certificate permit positive drive.
                    self._rejoin_context_longitudinal_constraint='braking_or_coasting'
                else:
                    self._coasting_measurement_time=None
            self._rejoin_last_report=report
            if report.get('clear',False):
                self._rejoin_context_approved=True
                return report
            if (report.get('reason')=='measured_rejoin_traffic_conflict'
                    and self.rejoin_configuration.get('replan_after_traffic_conflict',False)
                    and cached_coast_report is not None
                    and not cached_coast_report.get('clear',False)
                    and cached_coast_report.get('reason')=='measured_rejoin_traffic_conflict'):
                replacement_predecessor={
                    'route_id':(self._active_rejoin_provenance or {}).get('route_id'),
                    'start_progress_m':self._active_rejoin.start_progress_m,
                    'end_progress_m':self._active_rejoin.end_progress_m,
                    'revocation_reason':report.get('reason'),
                    'full_validation':report,'coasting_validation':cached_coast_report,
                    'revoked_before_replacement_build':True}
            self._cancel_rejoin('cached_rejoin_invalidated:'+str(report.get('reason')))
            if replacement_predecessor is None:
                return None
        nominal_here=float(line.nominal_offset_m(progress))
        nominal_heading=line.sample(progress)[1]
        actual_lateral=float(raw_action.get('_rejoin_actual_lateral_m',nominal_here))
        if (abs(actual_lateral-nominal_here)<=self.nominal_capture and
                abs(wrap_angle(nominal_heading-yaw))<=.20):
            return None
        cooldown=float(self.rejoin_configuration.get('attempt_interval_s',.20))
        if replacement_predecessor is None and self._rejoin_clock-self._last_rejoin_attempt_clock<cooldown:
            return None
        self._last_rejoin_attempt_clock=self._rejoin_clock
        effective=self.wheelbase+self.understeer_gradient*speed*speed
        entry_curvature=math.tan(self.previous_steer)/effective
        def checked_build(configuration):
            return build_rejoin(self.geometry_view(reference),self.geometry_view(line),line,
                start_progress_m=progress,ego_xy=np.asarray(position)[:2],ego_yaw=yaw,
                body_speed_mps=speed,entry_curvature_per_m=entry_curvature,
                road_rule=road_rule,observation=observation,configuration=configuration)
        route,report=checked_build(self.rejoin_configuration)
        full_traffic_rejections=sum(a.get('reason')=='measured_rejoin_traffic_conflict'
            for a in report.get('attempts',[]))
        initial_coasting=False
        if (route is None and not report.get('accepted',False) and full_traffic_rejections
                and self.rejoin_configuration.get('allow_initial_coasting_return',False)):
            route,report=checked_build(dict(self.rejoin_configuration,
                longitudinal_constraint='braking_or_coasting'))
            initial_coasting=bool(route is not None and report.get('accepted',False))
        audit_enabled=bool(self.rejoin_configuration.get('allow_initial_coasting_return',False)
            or self.rejoin_configuration.get('replan_after_traffic_conflict',False))
        if audit_enabled:
            report=dict(report,checked_planning_origin=(
                ('cached_traffic_replan_coast' if initial_coasting else 'cached_traffic_replan_full')
                if replacement_predecessor is not None else
                ('initial_coast_retry' if initial_coasting else 'initial_full')),
                revoked_predecessor=replacement_predecessor,
                full_drive_traffic_rejection_count_before_coasting=full_traffic_rejections,
                initial_coasting_certificate_used=initial_coasting)
        self._rejoin_last_report=report
        if route is not None and report.get('accepted',False):
            self._active_rejoin=route
            self._rejoin_context_approved=True
            if initial_coasting:
                self._rejoin_context_longitudinal_constraint='braking_or_coasting'
                self._coasting_measurement_time=float(observation.get(
                    '_measurement_time_s',self._rejoin_clock))
            if audit_enabled:
                self._rejoin_plan_serial+=1
                self._active_rejoin_provenance={
                    'route_id':f'rejoin-{self._rejoin_plan_serial:06d}',
                    'route_origin':report['checked_planning_origin'],
                    'predecessor_route_id':(replacement_predecessor or {}).get('route_id'),
                    'predecessor_cancel_reason':(replacement_predecessor or {}).get('revocation_reason'),
                    'approval_epoch_world_s':self._rejoin_clock,
                    'approval_measurement_time_s':float(observation.get('_measurement_time_s',self._rejoin_clock)),
                    'initial_longitudinal_constraint':self._rejoin_context_longitudinal_constraint,
                    'revoked_predecessor':replacement_predecessor,
                    'built_from_actual_pose':True}
            return report
        return None

    @staticmethod
    def is_nominal(action):
        return action.get('path_mode') == 'nominal_line' and action.get('path_profile_lane_m') is None

    @staticmethod
    def pursuit_lookahead(speed):
        return float(np.clip(4.5 + abs(speed)*.28, 5., 20.))

    def refresh_nominal_request(self, action, line, progress, speed):
        """Refresh only map-derived nominal geometry at50Hz, never ML outputs."""
        refreshed = dict(action)
        if line is not None and self.is_nominal(action):
            preview = progress + self.pursuit_lookahead(speed)
            refreshed['target_offset_m'] = line.nominal_offset_m(preview)
            refreshed['nominal_racing_line_offset_m'] = refreshed['target_offset_m']
            refreshed['nominal_geometry_refresh_progress_m'] = float(progress)
        return refreshed

    def nominal_route_context(self, reference, line, progress, position, yaw, speed, *,
                              raw_action, actual_lateral, observation, racing_parameters=None, road_rule=None, current_time_s=None):
        """Permit intrinsic mapped-line motion only after a measured route check.

        Longitudinal intervals include acceleration/braking uncertainty. Their
        lateral envelope samples the actual variable path, not a fixed lane.
        This supplements the unchanged final predictive traffic guard.
        """
        parameters=dict(racing_parameters or {})
        if self.rejoin_enabled:
            rejoin_request=dict(raw_action,_rejoin_actual_lateral_m=float(actual_lateral))
            rejoin_observation=dict(observation)
            if current_time_s is not None:
                self._rejoin_clock=float(current_time_s)
                measured_at=float(observation.get('_measurement_time_s',current_time_s))
                rejoin_observation['_elapsed_since_measurement_s']=max(0.,float(current_time_s)-measured_at)
            rejoin_report=self._rejoin_route_context(reference,line,progress,position,yaw,speed,
                rejoin_request,rejoin_observation,road_rule)
            if rejoin_report is not None and self._rejoin_context_approved:
                route=self._active_rejoin
                metadata={'eligible':True,'clear':True,'reason':'validated_finite_rejoin',
                    'opponent_ground_truth_used':False,'blocked_track_ids':[],
                    'rejoin_trajectory':rejoin_report,'route_start_progress_m':route.start_progress_m,
                    'route_end_progress_m':route.end_progress_m,
                    'current_rejoin_speed_cap_mps':route.minimum_speed(progress,self.preview_m),
                    'original_nominal_capture_threshold_unchanged':True}
                if self._active_rejoin_provenance is not None:
                    metadata['checked_route_provenance']=dict(self._active_rejoin_provenance)
                if self._rejoin_context_longitudinal_constraint is not None:
                    if self.coasting_brake_floor > route.brake_decel_mps2:
                        raise ValueError('Coasting brake floor exceeds the route declared braking capability')
                    metadata.update(longitudinal_constraint=self._rejoin_context_longitudinal_constraint,
                        required_braking_decel_mps2=self.coasting_brake_floor,positive_drive_permitted=False,
                        same_cached_geometry_retained_under_constraint=True)
                return parameters,metadata
        metadata={'eligible':False,'clear':False,'reason':'not_captured_nominal',
            'base_lane_rate_mps':float(parameters.get('lane_rate_mps',1.)),
            'opponent_ground_truth_used':False,'blocked_track_ids':[]}
        if line is None or not self.is_nominal(raw_action):
            return parameters,metadata
        error=float(actual_lateral-line.nominal_offset_m(progress))
        path_heading=line.sample(progress)[1]
        if abs(error)>self.nominal_capture or abs(wrap_angle(path_heading-yaw))>.20:
            return parameters,metadata
        metadata['eligible']=True
        horizon=float(self.config.get('nominal_route_horizon_s',2.))
        times=np.linspace(0.,horizon,21)
        lower=np.full(len(times),float(progress));upper=lower.copy();middle=lower.copy()
        metric_ratio=(np.asarray(line.ds)/np.asarray(reference.ds)) if (
            hasattr(line,'ds') and hasattr(line,'s') and hasattr(reference,'ds')) else None
        def metric(s):
            if metric_ratio is not None:
                return max(.15,float(np.interp(s%reference.length,line.s,metric_ratio)))
            before=line.sample(s-.5)[0];after=line.sample(s+.5)[0]
            return max(.15,float(np.linalg.norm(np.asarray(after)-np.asarray(before))))
        for i in range(1,len(times)):
            dt=times[i]-times[i-1];t=.5*(times[i]+times[i-1])
            lower[i]=lower[i-1]+max(0.,speed-reference.brake_decel*t)*dt/metric(lower[i-1])
            upper[i]=upper[i-1]+max(0.,speed+reference.acceleration*t)*dt/metric(upper[i-1])
            middle[i]=middle[i-1]+max(0.,speed)*dt/metric(middle[i-1])
        fractions=np.linspace(0.,1.,7)
        progress_grid=lower[:,None]+(upper-lower)[:,None]*fractions
        offsets=np.array([[line.nominal_offset_m(s) for s in group] for group in progress_grid])
        offsets+=error*np.exp(-times[:,None]/.6)
        low_lane=offsets.min(axis=1);high_lane=offsets.max(axis=1)
        half=float(parameters.get('ego_half_width_m',1.05))+float(parameters.get('lateral_margin_m',.1))
        uncertainty_cap=float(parameters.get('position_uncertainty_cap_m',.5))
        blockers=[]
        for row in observation.get('hazard_tracks',observation.get('confirmed_tracks',[])):
            distance=float(row['relative_distance_m']);velocity=float(row.get('speed_mps',0.))
            sigma=max(1.,float(row.get('speed_std_mps',2.)))
            vlow,vhigh=(velocity-sigma,velocity+sigma) if row.get('velocity_valid',True) else (0.,70.)
            hazard_low=distance+vlow*times;hazard_high=distance+vhigh*times
            possible_near=(hazard_low<(upper-progress)+8.5)&(hazard_high>(lower-progress)-8.5)
            center=float(row['lateral_m'])
            uncertainty=min(uncertainty_cap,max(.15,float(row.get('position_std_m',.35))))
            left=max(float(row.get('lateral_max_m',center+1.05)),center+1.05)+uncertainty
            right=min(float(row.get('lateral_min_m',center-1.05)),center-1.05)-uncertainty
            lateral_overlap=(right<high_lane+half)&(left>low_lane-half)
            if np.any(possible_near&lateral_overlap):
                blockers.append(row.get('track_id'))
        road_valid=True
        if road_rule is not None:
            wheel_points=[];wheel_progress=[]
            for index,(s,t) in enumerate(zip(middle,times)):
                point,heading,_,_=line.sample(s)
                point=np.asarray(point)+error*math.exp(-t/.6)*np.array([-math.sin(heading),math.cos(heading)])
                if index==0:
                    point,heading=np.asarray(position)[:2],yaw
                wheel_points.extend(road_rule.wheel_contact_positions(point,heading))
                wheel_progress.extend([s]*4)
            legal=road_rule.contains_points(np.asarray(wheel_points),progress_m=np.asarray(wheel_progress)).reshape(-1,4)
            road_valid=bool(np.all(legal.sum(axis=1)>=2))
        metadata.update({'blocked_track_ids':blockers,'future_road_two_contact_rule_valid':road_valid,
            'horizon_s':horizon,'trajectory_samples':len(times),'current_nominal_error_m':error})
        if blockers or not road_valid:
            metadata['reason']='measured_nominal_route_conflict' if blockers else 'future_nominal_road_contact_conflict'
            return parameters,metadata
        ahead=progress+self.pursuit_lookahead(speed)
        derivative=(line.nominal_offset_m(ahead+.5)-line.nominal_offset_m(ahead-.5))
        required=abs(derivative)*abs(speed)/metric(progress)
        floor=float(self.config.get('nominal_route_lane_rate_floor_mps',5.5))
        maximum=float(self.config.get('nominal_route_lane_rate_max_mps',8.))
        rate=max(metadata['base_lane_rate_mps'],floor,required*1.15+.5)
        metadata.update({'clear':True,'reason':'captured_mapped_nominal_route_clear',
            'intrinsic_nominal_rate_mps':required,
            'preview_offset_parameter_rate_mps':required,
            'preview_parameter_rate_above_diagnostic_limit':bool(rate>maximum),
            'suggested_preview_parameter_rate_mps':rate,
            'passing_lane_slew_rate_mps':metadata['base_lane_rate_mps'],
            'mapped_route_approval_uses_scalar_preview_rate':False})
        return parameters,metadata

    @staticmethod
    def offset_geometry(reference, progress, lane):
        point, heading, speed, curvature = reference.sample(progress)
        point = np.asarray(point) + float(lane)*np.array([-math.sin(heading),math.cos(heading)])
        denominator = 1. - float(lane)*curvature
        if denominator > .20:
            return point, heading, speed, curvature/denominator
        # A hypothetical wide lane may approach the parallel-curve cusp. It
        # must be rejected by route selection, but an unexpected actual state
        # still needs a finite guarded recovery command rather than an exception.
        def offset_point(s):
            p,h,_,_=reference.sample(s)
            return np.asarray(p)+float(lane)*np.array([-math.sin(h),math.cos(h)])
        step=2.
        before,after=offset_point(progress-step),offset_point(progress+step)
        first=(after-before)/(2.*step)
        second=(after-2.*point+before)/(step*step)
        norm=float(np.linalg.norm(first))
        local_heading=math.atan2(first[1],first[0]) if norm>.01 else heading
        local_curvature=float((first[0]*second[1]-first[1]*second[0])/max(norm**3,1e-6))
        return point,local_heading,speed,local_curvature

    def feedforward_curvature(self, reference, line, progress, lane, source, raw_curvature):
        """Average only feedforward curvature on a symmetric physical window.

        The original map progress can stretch/compress on the mapped apex path.
        Quadrature therefore uses that path's actual segment lengths, rather
        than treating original-map metres as driven metres. A fixed offset uses
        its positive parallel-curve metric. Current pose/heading feedback and
        final physical/traffic limits remain independent and unfiltered.
        """
        sigma = (self.continuous_reference_sigma if source=='guarded_constant_offset'
                 and getattr(reference,'continuous_geometry',False) else self.curvature_smoothing_sigma)
        metadata = {'enabled':sigma>0., 'used':False, 'sigma_m':sigma,
            'raw_curvature_per_m':float(raw_curvature),
            'filtered_curvature_per_m':float(raw_curvature),
            'symmetric_physical_window':True, 'temporal_filter_used':False,
            'feedback_filtered':False, 'speed_profile_modified':False}
        if getattr(line,'finite_domain',False):
            if sigma<=0.:
                metadata['reason']='disabled'
                return float(raw_curvature),metadata
            count=max(12,int(math.ceil(3.*sigma/.25)))
            offsets=np.linspace(-3.*sigma,3.*sigma,2*count+1)
            samples=np.array([line.advance_by_distance(progress,float(d)) for d in offsets])
            values=np.asarray(line.curvature_at(samples),float)
            if not np.isfinite(values).all():
                raise ValueError('Finite rejoin curvature quadrature is invalid')
            weights=np.exp(-.5*(offsets/sigma)**2)
            averaged=float(np.dot(weights,values)/weights.sum())
            metadata.update(used=True,reason='finite_route_physical_arc_gaussian',
                filtered_curvature_per_m=averaged,support_radius_m=3.*sigma,
                quadrature_count=len(offsets),finite_domain=True,
                quadrature_step_m=float(offsets[1]-offsets[0]))
            return averaged,metadata
        if sigma<=0.:
            metadata['reason']='disabled'
            return float(raw_curvature),metadata
        route = line if source=='variable_nominal_path' else reference
        if route is None or not all(hasattr(route,x) for x in ('s','ds','curvature','length')):
            metadata['reason']='route_has_no_physical_arc_table'
            return float(raw_curvature),metadata
        key=(id(route),source,float(lane) if source!='variable_nominal_path' else None)
        if key!=self._curvature_table_key:
            s=np.asarray(route.s,float)
            base_ds=np.asarray(getattr(route,'base_ds',route.ds),float)
            ds=np.asarray(route.ds,float).copy()
            curvature=np.asarray(route.curvature,float).copy()
            valid=np.ones(len(s),bool)
            if source!='variable_nominal_path':
                denominator=1.-float(lane)*curvature
                valid=denominator>.20
                # Invalid remote segments do not break monotonic arc lookup;
                # any window touching one is separately refused below.
                if getattr(route,'continuous_geometry',False):
                    offset_table=route.offset_arc_table(lane)
                    ds=np.asarray(offset_table['ds'],float).copy()
                    curvature=np.asarray(offset_table['curvature'],float).copy()
                    valid=np.asarray(offset_table['valid'],bool).copy()
                else:
                    ds=base_ds*np.maximum(denominator,.20)
                    curvature=curvature/np.maximum(denominator,.20)
            if (len(s)<3 or s.shape!=ds.shape or s.shape!=curvature.shape or
                    s.shape!=base_ds.shape or not np.isfinite(np.r_[s,ds,base_ds,curvature]).all()
                    or np.any(ds<=0.) or np.any(base_ds<=0.)):
                metadata['reason']='invalid_physical_arc_table'
                return float(raw_curvature),metadata
            arc=np.r_[0.,np.cumsum(ds[:-1])]
            length=float(ds.sum())
            self._curvature_table=(s,base_ds,ds,arc,length,np.r_[arc,length],
                np.r_[curvature,curvature[0]],valid)
            self._curvature_table_key=key
        s,base_ds,ds,arc,length,extended_arc,extended_curvature,valid=self._curvature_table
        wrapped=float(progress)%float(route.length)
        index=min(max(int(np.searchsorted(s,wrapped,side='right')-1),0),len(s)-1)
        fraction=float(np.clip((wrapped-s[index])/base_ds[index],0.,1.))
        center=arc[index]+fraction*ds[index]
        support=3.*sigma
        half_count=max(12,int(math.ceil(support/.25)))
        offsets=np.linspace(-support,support,2*half_count+1)
        locations=(center+offsets)%length
        ids=np.clip(np.searchsorted(arc,locations,side='right')-1,0,len(s)-1)
        # Interpolation touches both vertices; include the next one as well.
        if not np.all(valid[ids]&valid[(ids+1)%len(s)]):
            metadata['reason']='window_touches_parallel_curve_cusp'
            return float(raw_curvature),metadata
        weights=np.exp(-.5*(offsets/sigma)**2)
        curvature=np.interp(locations,extended_arc,extended_curvature)
        averaged=float(np.dot(weights,curvature)/weights.sum())
        metadata.update(used=True,reason='symmetric_physical_arc_gaussian',
            filtered_curvature_per_m=averaged, support_radius_m=support,
            quadrature_count=len(offsets),physical_arc_center_m=float(center),
            quadrature_step_m=float(offsets[1]-offsets[0]))
        return averaged,metadata

    def control(self, reference, line, progress, position, yaw, speed, dt, *,
                raw_action, guarded_lane, actual_lateral, guard_flags=(),mapped_nominal_route_approved=False,
                longitudinal_decel_floor_mps2=0.):
        if dt <= 0 or not math.isfinite(dt):
            raise ValueError('Tracking timestep must be finite and positive')
        if self.yaw_stability_enabled:
            self._yaw_step_previous_steer=float(self.previous_steer)
        if self.rejoin_enabled:
            self._rejoin_clock += float(dt)
        geometry_reference = self.geometry_view(reference)
        continuous_reference = geometry_reference
        geometry_line = line  # The calibrated captured nominal path stays unchanged.
        continuous_status = None
        continuous_used = self.continuous_geometry_enabled
        if continuous_used:
            continuous_status = geometry_reference.parallel_status(progress, guarded_lane,
                preview_m=max(20., self.pursuit_lookahead(speed)))
            if not continuous_status['trackable']:
                # Preserve legacy finite recovery at an unexpected folded
                # parallel curve; do not pretend the new geometry is valid.
                geometry_reference = reference
                continuous_used = False
        desired = float(raw_action['target_offset_m'])
        nominal = bool(line is not None and self.is_nominal(raw_action))
        nominal_here = line.nominal_offset_m(progress) if nominal else None
        scalar_approved = abs(guarded_lane-desired) <= self.guard_tolerance
        approved = scalar_approved or (nominal and bool(mapped_nominal_route_approved))
        conflict = any(flag in guard_flags for flag in ['predicted_lane_conflict',
            'alongside_lane_occupied','mapped_nominal_route_conflict'])
        captured = nominal and abs(actual_lateral-nominal_here) <= self.nominal_capture
        if self.rejoin_enabled and (not nominal or conflict):
            self._cancel_rejoin('final_guard_or_policy_revoked_rejoin')
        rejoin_route=(self._active_rejoin if nominal and mapped_nominal_route_approved
            and self._rejoin_context_approved and not conflict else None)
        lane_rate = 0. if self.previous_lane is None else (guarded_lane-self.previous_lane)/dt
        if rejoin_route is not None:
            geometry = rejoin_route.sample(progress)
            source = 'validated_rejoin_path'
            direct = True
        elif nominal and approved and captured and not conflict:
            geometry = geometry_line.sample(progress)
            source = 'variable_nominal_path'
            direct = True
        else:
            geometry = self.offset_geometry(geometry_reference, progress, guarded_lane)
            source = 'guarded_constant_offset'
            direct = self.previous_lane is not None and abs(lane_rate) < .10 and not conflict
        route_status=constant_offset_route_status(reference,progress,guarded_lane,preview_m=0.)
        if source=='guarded_constant_offset' and not route_status['trackable']:
            direct=False
        if self.steering_method=='pure_pursuit':
            direct=False
        point, heading, _, curvature = geometry
        normal = np.array([-math.sin(heading),math.cos(heading)])
        cross_error = float(np.dot(np.asarray(point)-np.asarray(position)[:2],normal))
        heading_error = wrap_angle(heading-yaw)
        effective_wheelbase=self.wheelbase+self.understeer_gradient*speed*speed
        if direct:
            feedforward_curvature,curvature_smoothing = self.feedforward_curvature(
                geometry_reference,rejoin_route if rejoin_route is not None else geometry_line,progress,guarded_lane,source,curvature)
        else:
            feedforward_curvature=curvature
            curvature_smoothing={'enabled':self.curvature_smoothing_sigma>0., 'used':False,
                'sigma_m':self.curvature_smoothing_sigma,
                'raw_curvature_per_m':float(curvature), 'filtered_curvature_per_m':float(curvature),
                'reason':'feedforward_not_used_by_selected_method', 'symmetric_physical_window':True,
                'temporal_filter_used':False, 'feedback_filtered':False, 'speed_profile_modified':False}
        feedforward = math.atan(effective_wheelbase*feedforward_curvature)
        feedback = self.heading_gain*heading_error + math.atan2(self.lateral_gain*cross_error,abs(speed)+self.speed_softening)
        nominal_pursuit=(source=='variable_nominal_path' and self.steering_method=='pure_pursuit')
        rejoin_pursuit=(rejoin_route is not None and self.steering_method=='pure_pursuit')
        if nominal_pursuit or rejoin_pursuit:
            target=(np.asarray(rejoin_route.sample(rejoin_route.advance_by_distance(progress,self.pursuit_lookahead(speed)))[0])
                if rejoin_pursuit else np.asarray(geometry_line.sample(progress+self.pursuit_lookahead(speed))[0]))
            delta=target-np.asarray(position)[:2]
            distance=max(.5,float(np.linalg.norm(delta)))
            alpha=wrap_angle(math.atan2(delta[1],delta[0])-yaw)
            unconstrained=math.atan2(2.*effective_wheelbase*math.sin(alpha),distance)
            clipped=float(np.clip(unconstrained,-self.max_steer,self.max_steer))
            steering=float(np.clip(clipped,self.previous_steer-self.steer_rate*dt,
                self.previous_steer+self.steer_rate*dt))
        elif direct:
            unconstrained = feedforward+feedback
            clipped = float(np.clip(unconstrained,-self.max_steer,self.max_steer))
            steering = float(np.clip(clipped,self.previous_steer-self.steer_rate*dt,
                self.previous_steer+self.steer_rate*dt))
        else:
            # During guarded lateral slewing, keep the approved target exactly.
            # A nominal path point cannot bypass an occupied/blocked lane.
            self.fallback.previous_steer = self.previous_steer
            self.fallback.wheelbase = effective_wheelbase
            steering,_ = self.fallback.control(geometry_reference,progress,position,yaw,speed,dt,lateral_offset=guarded_lane)
            unconstrained = steering
            clipped = steering
        self.previous_steer = steering
        self.previous_lane = float(guarded_lane)
        longitudinal_accel=0. if self.previous_speed is None else (speed-self.previous_speed)/dt
        self.previous_speed=float(speed)
        demand_metadata=None
        longitudinal_demand=longitudinal_accel
        if self.longitudinal_friction_estimator_enabled:
            longitudinal_demand,demand_metadata=self.longitudinal_friction_demand(
                longitudinal_accel,dt,minimum_decel_mps2=longitudinal_decel_floor_mps2)
        friction=float(self.config.get('friction_coefficient',1.5))*9.81*.95
        available=math.sqrt(max(0.,friction*friction-longitudinal_demand*longitudinal_demand))
        lateral_budget=float(getattr(reference,'lateral_accel',8.))
        corrective_budget=lateral_budget*float(self.config.get('steering_lateral_reserve_factor',1.2))
        allowed_lateral=min(corrective_budget,available)
        speed_steer_limit=min(self.max_steer,math.atan(effective_wheelbase*allowed_lateral/max(speed*speed,25.)))
        before_stability=steering
        if self.kinematic_steering_limit_enabled:
            steering=float(np.clip(steering,-speed_steer_limit,speed_steer_limit))
        self.previous_steer=steering
        metadata={'version':self.VERSION,'geometry_source':source,
            'steering_method':'rejoin_path_pure_pursuit' if rejoin_pursuit else ('variable_path_pure_pursuit' if nominal_pursuit else (
                'curvature_feedforward_feedback' if direct else 'guarded_pure_pursuit_transition')),
            'nominal_request_refreshed_at_physics_rate':nominal,'nominal_capture':bool(captured),
            'guard_approved_nominal_preview':bool(scalar_approved),'guard_conflict':bool(conflict),
            'mapped_nominal_route_approval_supplied':bool(mapped_nominal_route_approved),
            'mapped_nominal_route_approval_used':bool(nominal and captured and mapped_nominal_route_approved and not conflict),
            'guarded_lane_rate_mps':lane_rate,'selected_path_point_xy_m':np.asarray(point).tolist(),
            'constant_offset_geometry_status':route_status,
            'selected_path_heading_rad':float(heading),'selected_path_curvature_per_m':float(curvature),
            'selected_path_cross_error_m':cross_error,'selected_path_heading_error_rad':heading_error,
            'steering_feedforward_rad':feedforward,'steering_feedback_rad':feedback,
            'feedforward_curvature_smoothing':curvature_smoothing,
            'steering_unconstrained_rad':unconstrained,'steering_command_rad':steering,
            'steering_before_lateral_budget_rad':before_stability,
            'understeer_gradient_s2_per_m':self.understeer_gradient,
            'effective_steering_wheelbase_m':effective_wheelbase,
            'steering_kinematic_limit_enabled':self.kinematic_steering_limit_enabled,
            'kinematic_lateral_demand_is_measured_tire_force':False,
            'steering_lateral_budget_limited':abs(steering-before_stability)>1e-9,
            'steering_speed_dependent_limit_rad':speed_steer_limit,
            'commanded_lateral_accel_before_limit_mps2':speed*speed*abs(math.tan(before_stability))/self.wheelbase,
            'uncalibrated_kinematic_lateral_demand_mps2':speed*speed*abs(math.tan(before_stability))/self.wheelbase,
            'estimated_body_lateral_accel_before_limit_mps2':speed*speed*abs(math.tan(before_stability))/effective_wheelbase,
            'estimated_body_lateral_accel_command_mps2':speed*speed*abs(math.tan(steering))/effective_wheelbase,
            'body_lateral_accel_estimate_is_measured_force':False,
            'commanded_lateral_accel_limit_mps2':allowed_lateral if self.kinematic_steering_limit_enabled else None,
            'steering_kinematic_limit_diagnostic_mps2':allowed_lateral,
            'measured_longitudinal_accel_mps2':longitudinal_accel,
            'steering_rate_limited':abs(steering-clipped)>1e-9,
            'actual_lateral_m':float(actual_lateral),'guarded_target_offset_m':float(guarded_lane),
            'requested_preview_offset_m':desired,
            'nominal_offset_here_m':nominal_here,'opponent_ground_truth_used':False}
        if self.continuous_geometry_enabled:
            metadata['continuous_reference_geometry'] = {
                'enabled':True,'used':bool(continuous_used and source!='variable_nominal_path'),
                'version':ContinuousGeometry.VERSION,'interpolation':('original calibrated nominal geometry' if source=='variable_nominal_path' else 'periodic C2 XY spline'),
                'original_nominal_geometry_retained':bool(source=='variable_nominal_path'),
                'position_tangent_curvature_from_same_curve':bool(continuous_used and source!='variable_nominal_path'),
                'feedback_or_actuator_filtered':False,'legacy_speed_profiles_retained':True,
                'parallel_curve_status':continuous_status,
                'speed_cap_mps':(rejoin_route.minimum_speed(progress,self.preview_m) if rejoin_route is not None else (geometry_line.minimum_speed(progress,self.preview_m) if source=='variable_nominal_path' else continuous_reference.offset_minimum_speed(progress,self.preview_m,guarded_lane))),
                'legacy_cusp_recovery_used':bool(not continuous_used and source!='variable_nominal_path')}
        if self.rejoin_enabled:
            metadata['validated_rejoin_approval_used']=rejoin_route is not None
            metadata['rejoin_trajectory']={'enabled':True,'active':rejoin_route is not None,
                'context_approved':self._rejoin_context_approved,'latest_check':self._rejoin_last_report,
                'start_progress_m':None if rejoin_route is None else rejoin_route.start_progress_m,
                'end_progress_m':None if rejoin_route is None else rejoin_route.end_progress_m}
            if rejoin_route is not None and self._active_rejoin_provenance is not None:
                metadata['checked_route_provenance']=dict(self._active_rejoin_provenance)
            if rejoin_route is not None:
                metadata['continuous_reference_geometry'].update(interpolation='validated finite C2 rejoin',used=True,
                    position_tangent_curvature_from_same_curve=True,original_nominal_geometry_retained=False,
                    legacy_cusp_recovery_used=False,parallel_curve_status_is_rejoin_validity=False)
        if demand_metadata is not None:
            metadata['longitudinal_friction_demand_estimator']=demand_metadata
        return steering,metadata

    def longitudinal_friction_demand(self, raw_accel, dt, *, minimum_decel_mps2=0.):
        """Causal robust body-acceleration estimate, with an immediate floor.

        This is a software tire-demand proxy, not a measured tire force. No
        steering or feedback signal is filtered. Persistent large deceleration
        remains large; it is never clipped down to a convenient planning limit.
        """
        raw_accel,dt,minimum_decel_mps2=map(float,(raw_accel,dt,minimum_decel_mps2))
        if (not all(math.isfinite(x) for x in (raw_accel,dt,minimum_decel_mps2))
                or dt<=0. or minimum_decel_mps2<0.):
            raise ValueError('Finite acceleration, positive dt and nonnegative mandatory deceleration required')
        self._longitudinal_accel_samples.append(raw_accel)
        median=float(np.median(self._longitudinal_accel_samples))
        if self._filtered_longitudinal_accel is None:
            self._filtered_longitudinal_accel=median
        else:
            weight=-math.expm1(-dt/self.longitudinal_accel_filter_tau)
            self._filtered_longitudinal_accel+=weight*(median-self._filtered_longitudinal_accel)
        # Narrow this correction to negative derivative notches. Positive
        # measured acceleration remains an instantaneous conservative floor;
        # powered acceleration can never acquire a larger steering allowance
        # by being hidden behind the body-signal filter.
        positive_floor=max(raw_accel,0.)
        demand=max(abs(self._filtered_longitudinal_accel),positive_floor,minimum_decel_mps2)
        return demand,{'enabled':True,'raw_body_acceleration_mps2':raw_accel,
            'median_body_acceleration_mps2':median,
            'filtered_body_acceleration_mps2':self._filtered_longitudinal_accel,
            'filter_tau_s':self.longitudinal_accel_filter_tau,
            'median_window_samples':self._longitudinal_accel_samples.maxlen,
            'available_samples':len(self._longitudinal_accel_samples),
            'instant_minimum_decel_floor_mps2':minimum_decel_mps2,
            'instant_positive_acceleration_floor_mps2':positive_floor,
            'longitudinal_tire_demand_proxy_mps2':demand,
            'actual_tire_force_measured':False,'braking_command_modified':False,
            'steering_or_feedback_filtered':False,'persistent_deceleration_capped':False}

    def _yaw_sideslip_feedback(self, steering, speed, metadata, *, brake, minimum_decel,
            measured_yaw_rate_rps, measured_sideslip_rad, dt):
        if not self.yaw_stability_enabled:
            return steering,None
        try:
            yaw_rate,slip,period=map(float,(measured_yaw_rate_rps,measured_sideslip_rad,dt))
            curvature=float(metadata['selected_path_curvature_per_m'])
        except (TypeError,ValueError,KeyError):
            self._yaw_previous_sideslip=None
            return steering,None
        if (not all(math.isfinite(x) for x in (yaw_rate,slip,period,curvature))
                or not 0.<period<=.2 or abs(slip)>math.pi):
            self._yaw_previous_sideslip=None
            return steering,None
        previous=self._yaw_previous_sideslip
        self._yaw_previous_sideslip=slip
        growth=(0. if previous is None else np.sign(slip)*wrap_angle(slip-previous)/period)
        desired=curvature*speed
        direction=float(np.sign(yaw_rate))
        excess=direction*(yaw_rate-desired)
        config=self.yaw_stability_configuration
        curve=(metadata.get('geometry_source') in ('validated_rejoin_path','variable_nominal_path')
            or (metadata.get('steering_method')=='curvature_feedforward_feedback'
                and abs(float(metadata.get('guarded_lane_rate_mps',math.inf)))<.10))
        eligible=bool(curve and not metadata.get('guard_conflict',False))
        active=bool(eligible and speed>=config['minimum_speed_mps']
            and (brake>=config['minimum_brake_fraction'] or minimum_decel>0.)
            and (yaw_rate*desired>0. or abs(desired)<.02)
            and excess>config['yaw_rate_deadband_rps']
            and abs(slip)>config['sideslip_deadband_rad'] and slip*yaw_rate<0.
            and previous is not None and growth>config['sideslip_growth_deadband_rps'])
        correction=0.
        if active:
            effective=self.wheelbase+self.understeer_gradient*speed*speed
            yaw_correction=-direction*config['yaw_rate_gain']*effective/max(speed,1.)*(excess-config['yaw_rate_deadband_rps'])
            slip_correction=config['sideslip_gain']*slip
            correction=float(np.clip(yaw_correction+slip_correction,
                -config['maximum_correction_rad'],config['maximum_correction_rad']))
        return steering+correction,{'enabled':True,'active':active,
            'measured_yaw_rate_rps':yaw_rate,'measured_sideslip_rad':slip,
            'sideslip_definition':'velocity heading minus body heading',
            'desired_path_yaw_rate_rps':desired,'same_direction_yaw_excess_rps':excess,
            'outward_sideslip_growth_rps':float(growth),'explicit_curve_eligible':eligible,
            'steering_correction_requested_rad':correction,
            'yaw_rate_times_speed_is_tire_force':False,'braking_command_modified':False,
            'temporal_command_filter_used':False,'original_final_budget_has_priority':True}

    def apply_commanded_longitudinal_steering_limit(self, steering, body_speed, metadata, *,
            brake, minimum_decel_mps2=0.,mass=880.,wheel_radius=.35,brake_coefficient=700.,
            measured_yaw_rate_rps=None,measured_sideslip_rad=None,dt=None):
        """Final recap AFTER actual stability and mandatory braking overrides.

        Four equal brake torques use the same coefficient/mass/radius model as
        the existing controller. A new brake request tightens the limit now,
        even before a speed response enters the acceleration estimator. This
        helper never changes braking, raises a limit, or averages a command.
        """
        if not self.longitudinal_friction_estimator_enabled:
            return steering,metadata
        steering,body_speed,brake,minimum_decel_mps2,mass,wheel_radius,brake_coefficient=map(float,
            (steering,body_speed,brake,minimum_decel_mps2,mass,wheel_radius,brake_coefficient))
        if (not all(math.isfinite(x) for x in (steering,body_speed,brake,minimum_decel_mps2,mass,wheel_radius,brake_coefficient))
                or body_speed<0. or not 0.<=brake<=1. or minimum_decel_mps2<0.
                or min(mass,wheel_radius,brake_coefficient)<=0.):
            raise ValueError('Finite command/speed, brake in[0,1], nonnegative required deceleration and positive vehicle parameters required')
        if self._filtered_longitudinal_accel is None:
            raise RuntimeError('Call control before the post-command steering recap')
        command_decel=4.*brake_coefficient*brake/(mass*wheel_radius)
        pre=metadata.get('longitudinal_friction_demand_estimator',{})
        prior_demand=float(pre.get('longitudinal_tire_demand_proxy_mps2',abs(self._filtered_longitudinal_accel)))
        demand=max(abs(self._filtered_longitudinal_accel),prior_demand,command_decel,minimum_decel_mps2)
        friction=float(self.config.get('friction_coefficient',1.5))*9.81*.95
        remaining=math.sqrt(max(0.,friction*friction-demand*demand))
        allowed=min(float(metadata['commanded_lateral_accel_limit_mps2']),remaining)
        effective=self.wheelbase+self.understeer_gradient*body_speed*body_speed
        limit=min(float(metadata['steering_speed_dependent_limit_rad']),self.max_steer,
            math.atan(effective*allowed/max(body_speed*body_speed,25.)))
        command=float(np.clip(steering,-limit,limit))
        base_budget_limited=abs(command-steering)>1e-9
        proposal,yaw_stability=self._yaw_sideslip_feedback(command,body_speed,metadata,
            brake=brake,minimum_decel=minimum_decel_mps2,
            measured_yaw_rate_rps=measured_yaw_rate_rps,
            measured_sideslip_rad=measured_sideslip_rad,dt=dt)
        if yaw_stability is not None and yaw_stability['active']:
            previous=(self.previous_steer if self._yaw_step_previous_steer is None
                else self._yaw_step_previous_steer)
            rate_proposal=float(np.clip(proposal,previous-self.steer_rate*float(dt),
                previous+self.steer_rate*float(dt)))
            final=float(np.clip(rate_proposal,-limit,limit))
            yaw_stability.update(previous_actual_steering_rad=float(previous),
                original_final_command_rad=command,steering_before_final_budget_rad=rate_proposal,
                rate_limited=abs(rate_proposal-proposal)>1e-9,
                final_budget_limited=abs(final-rate_proposal)>1e-9,
                applied_steering_correction_rad=final-command,
                hard_budget_may_reduce_steer_instantly=True)
            command=final
        if yaw_stability is not None:
            yaw_stability.update(final_steering_budget_rad=limit,
                actual_brake_fraction=brake,minimum_required_decel_mps2=minimum_decel_mps2)
        result=dict(metadata)
        if yaw_stability is not None:
            result['yaw_sideslip_stability']=yaw_stability
        result.update(steering_command_rad=command,steering_speed_dependent_limit_rad=limit,
            commanded_lateral_accel_limit_mps2=allowed,
            steering_kinematic_limit_diagnostic_mps2=allowed,
            steering_lateral_budget_limited=bool(metadata['steering_lateral_budget_limited'] or base_budget_limited
                or (yaw_stability is not None and yaw_stability.get('final_budget_limited',False))),
            estimated_body_lateral_accel_command_mps2=body_speed*body_speed*abs(math.tan(command))/effective,
            longitudinal_friction_post_command={'applied':True,'actual_brake_fraction':brake,
                'commanded_brake_decel_proxy_mps2':command_decel,'minimum_required_decel_mps2':minimum_decel_mps2,
                'longitudinal_tire_demand_proxy_mps2':demand,'remaining_lateral_budget_mps2':allowed,
                'friction_acceleration_bound_mps2':friction,
                'brake_model':'four equal brake torques / mass / wheel radius',
                'mass_kg':mass,'wheel_radius_m':wheel_radius,'brake_coefficient_nm':brake_coefficient,
                'braking_command_modified':False,'limit_relaxed':False,
                'post_command_cap_limited':bool(base_budget_limited or
                    (yaw_stability is not None and yaw_stability.get('final_budget_limited',False)))})
        self.previous_steer=command
        return command,result

    def recovery_speed(self, target_speed, metadata):
        factor = float(np.clip(1.-.06*abs(metadata['selected_path_cross_error_m']),.55,1.))
        target = float(target_speed)*factor
        heading_recovery = abs(metadata['selected_path_heading_error_rad']) > .4
        if heading_recovery:
            target = min(target,12.)
        result = {'path_tracking_speed_factor':factor,'selected_path_heading_recovery':heading_recovery}
        if self.continuous_geometry_enabled:
            cap=float(metadata['continuous_reference_geometry']['speed_cap_mps'])
            target=min(target,cap)
            result['continuous_geometry_speed_cap_mps']=cap
            result['continuous_geometry_cap_tightens_only']=True
        return target,result

    def profile_acceleration(self, reference, line, progress, speed, *, raw_action,
                             actual_lateral, aggressive, current_cap, target_speed, guard_flags=(),tracking_metadata=None):
        if not self.config.get('profile_acceleration_feedforward',True):
            return 0.
        if (self.rejoin_enabled and self._active_rejoin is not None and
                (tracking_metadata or {}).get('geometry_source')=='validated_rejoin_path'):
            route=self._active_rejoin
            distance=float(np.clip(abs(speed)*.12,2.,6.))
            future=route.advance_by_distance(progress,distance)
            current_cap=min(float(current_cap),route.minimum_speed(progress,self.preview_m))
            future_cap=route.minimum_speed(future,self.preview_m)
            acceleration=(future_cap**2-current_cap**2)/(2.*max(.05,route.physical_distance(progress,future)))
            if acceleration>0. and (target_speed<current_cap-.75 or any(flag in guard_flags
                    for flag in ['predictive_stopping_gap','stopping_brake_feedforward'])):
                return 0.
            return float(np.clip(acceleration,-reference.brake_decel,reference.acceleration))
        step = float(np.clip(abs(speed)*.12,2.,6.))
        variable_captured=line is not None and (tracking_metadata or {}).get('geometry_source')=='variable_nominal_path' and not any(
                flag in guard_flags for flag in ['predicted_lane_conflict','mapped_nominal_route_conflict'])
        if variable_captured:
            # The vehicle's CURRENT lateral coordinate is not a future pose.
            # Query the captured variable trajectory's future profile directly.
            future_cap=line.minimum_speed(progress+step,self.preview_m)
        elif line is None:
            future_cap = reference.minimum_speed(progress+step,self.preview_m)
        elif aggressive is not None:
            future_cap,_ = aggressive.guard_profile_cap(progress+step,self.preview_m,raw_action,
                actual_lateral,line.minimum_speed(progress+step,self.preview_m))
        else:
            future_cap = line.minimum_speed(progress+step,self.preview_m)
        if (self.continuous_geometry_enabled and not variable_captured
                and tracking_metadata is not None):
            lane=float(tracking_metadata['guarded_target_offset_m'])
            geometry=self.geometry_view(reference)
            current_cap=min(float(current_cap),geometry.offset_minimum_speed(progress,self.preview_m,lane))
            future_cap=min(float(future_cap),geometry.offset_minimum_speed(progress+step,self.preview_m,lane))
            table=geometry.offset_arc_table(lane)
            arc=np.r_[0.,np.cumsum(np.asarray(table['ds'],float)[:-1])]
            physical_length=float(np.sum(table['ds']))
            def physical_coordinate(s):
                cycle=math.floor(float(s)/geometry.length)
                return cycle*physical_length+float(np.interp(float(s)%geometry.length,
                    np.r_[geometry.s,geometry.length],np.r_[arc,physical_length]))
            driven_distance=max(.05,physical_coordinate(progress+step)-physical_coordinate(progress))
            acceleration=(float(future_cap)**2-float(current_cap)**2)/(2.*driven_distance)
            if acceleration>0. and (target_speed<current_cap-.75 or any(flag in guard_flags
                    for flag in ['predictive_stopping_gap','stopping_brake_feedforward'])):
                return 0.
            return float(np.clip(acceleration,-reference.brake_decel,reference.acceleration))
        distances=[]
        if line is not None and (variable_captured or raw_action.get('path_profile_lane_m') is None):
            distances.append(float(np.linalg.norm(np.asarray(line.sample(progress+step)[0])-
                np.asarray(line.sample(progress)[0]))))
        if not variable_captured:
            lanes=[float(actual_lateral)]
            if raw_action.get('path_profile_lane_m') is not None:
                lanes.append(float(raw_action['path_profile_lane_m']))
            for lane in lanes:
                before=self.offset_geometry(reference,progress,lane)[0]
                after=self.offset_geometry(reference,progress+step,lane)[0]
                distances.append(float(np.linalg.norm(after-before)))
        # Profiles propagate on physical driven distance, not the original-map
        # parameter. The short-step chord is a conservative braking denominator.
        driven_distance=max(.05,min(distances))
        acceleration = (float(future_cap)**2-float(current_cap)**2)/(2.*driven_distance)
        # A lower recovery/traffic target must not disable required map-profile
        # braking. Suppress only POSITIVE feedforward under another constraint.
        if acceleration>0. and (target_speed<current_cap-.75 or any(flag in guard_flags
                for flag in ['predictive_stopping_gap','stopping_brake_feedforward'])):
            return 0.
        return float(np.clip(acceleration,-reference.brake_decel,reference.acceleration))

    def stability_actuators(self,throttle,brake,target_speed,body_speed,forward_speed,metadata,*,
                            mass=880.,wheel_radius=.35,brake_coefficient=700.,max_decel=8.):
        ratio=float(forward_speed)/max(float(body_speed),.1)
        unstable=body_speed>3. and (ratio<math.cos(.35) or abs(metadata['selected_path_heading_error_rad'])>.4)
        decel=0.
        if unstable:
            throttle=0.
            decel=min(max_decel,max(2.,1.2*(body_speed-min(target_speed,8.))))
            brake=max(brake,min(1.,decel*mass*wheel_radius/(4.*brake_coefficient)))
        gross_hold=bool(self.gross_heading_hold_enabled and body_speed<=5.
            and abs(metadata['selected_path_heading_error_rad'])>1.2)
        if gross_hold:
            # A stopped car pointing away from its route cannot safely solve
            # recovery by repeatedly reapplying throttle below the old3m/s gate.
            # This is a failed-closed state, never a successful lap outcome.
            throttle,brake=0.,1.
        result={'spin_recovery_active':bool(unstable),
            'forward_speed_ratio':ratio,'physical_speed_for_braking_mps':float(body_speed),
            'minimum_stability_decel_mps2':decel}
        if self.gross_heading_hold_enabled:
            result.update(gross_heading_hold_active=gross_hold,
                controller_recovery_failed_closed=gross_hold,
                gross_heading_hold_is_success=False)
        return throttle,brake,result

    def longitudinal(self,target_speed,body_speed,wheel_speed,*,profile_accel=0.,mass=880.,
                     wheel_radius=.35,brake_coefficient=700.,max_accel=5.,max_decel=8.):
        if target_speed <= .15:
            return longitudinal_control(target_speed,body_speed,wheel_speed,mass=mass,
                wheel_radius=wheel_radius,brake_coefficient=brake_coefficient,
                max_accel=max_accel,max_decel=max_decel)
        gain=float(self.config.get('speed_feedback_gain',1.2))
        acceleration=float(np.clip(profile_accel+gain*(target_speed-body_speed),-max_decel,max_accel))
        if acceleration >= -.3:
            throttle=float(np.clip(.12+.18*max(0.,acceleration),0.,1.))
            slip=wheel_speed-body_speed-max(3.,.16*body_speed)
            if slip>0:throttle*=max(.08,1./(1.+.3*slip))
            return throttle,0.
        return 0.,float(np.clip(-acceleration*mass*wheel_radius/(4.*brake_coefficient),0.,1.))
