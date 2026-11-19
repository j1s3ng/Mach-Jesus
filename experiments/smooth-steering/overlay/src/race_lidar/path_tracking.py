"""Opt-in selected-path tracking and scheduled-speed control.

All geometry uses the supplied original progress and map, plus measured ego
pose. Passing approval remains the caller's measured-traffic guard decision.
"""
from __future__ import annotations

import math
import numpy as np
from race_lidar.hotlap_driver import PurePursuit, longitudinal_control, wrap_angle


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
        if not math.isfinite(self.curvature_smoothing_sigma) or not 0.<=self.curvature_smoothing_sigma<=4.:
            raise ValueError('feedforward_curvature_smoothing_sigma_m must be finite within[0,4]')
        # These cache map-derived quadrature only; there is no temporal filter
        # state in the feedback, avoidance target, or actuator command.
        self._curvature_table_key = None
        self._curvature_table = None
        if not math.isfinite(self.understeer_gradient) or not 0.<=self.understeer_gradient<=.005:
            raise ValueError('understeer_gradient_s2_per_m must be finite within[0,.005]')
        if self.steering_method not in {'curvature_feedback','pure_pursuit'}:
            raise ValueError('steering_method must be curvature_feedback or pure_pursuit')
        self.previous_steer = 0.
        self.previous_lane = None
        self.previous_speed = None
        self.fallback = PurePursuit(wheelbase=wheelbase, max_steer=self.max_steer,
            steer_rate=self.steer_rate)
        values = [self.wheelbase,self.max_steer,self.steer_rate,self.heading_gain,
            self.lateral_gain,self.speed_softening,self.nominal_capture,self.guard_tolerance]
        if not all(math.isfinite(x) and x > 0 for x in values) or not math.isfinite(self.preview_m) or self.preview_m < 0:
            raise ValueError('Tracking parameters must be positive and preview finite/nonnegative')

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
                              raw_action, actual_lateral, observation, racing_parameters=None, road_rule=None):
        """Permit intrinsic mapped-line motion only after a measured route check.

        Longitudinal intervals include acceleration/braking uncertainty. Their
        lateral envelope samples the actual variable path, not a fixed lane.
        This supplements the unchanged final predictive traffic guard.
        """
        parameters=dict(racing_parameters or {})
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
        sigma = self.curvature_smoothing_sigma
        metadata = {'enabled':sigma>0., 'used':False, 'sigma_m':sigma,
            'raw_curvature_per_m':float(raw_curvature),
            'filtered_curvature_per_m':float(raw_curvature),
            'symmetric_physical_window':True, 'temporal_filter_used':False,
            'feedback_filtered':False, 'speed_profile_modified':False}
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
                raw_action, guarded_lane, actual_lateral, guard_flags=(),mapped_nominal_route_approved=False):
        if dt <= 0 or not math.isfinite(dt):
            raise ValueError('Tracking timestep must be finite and positive')
        desired = float(raw_action['target_offset_m'])
        nominal = bool(line is not None and self.is_nominal(raw_action))
        nominal_here = line.nominal_offset_m(progress) if nominal else None
        scalar_approved = abs(guarded_lane-desired) <= self.guard_tolerance
        approved = scalar_approved or (nominal and bool(mapped_nominal_route_approved))
        conflict = any(flag in guard_flags for flag in ['predicted_lane_conflict',
            'alongside_lane_occupied','mapped_nominal_route_conflict'])
        captured = nominal and abs(actual_lateral-nominal_here) <= self.nominal_capture
        lane_rate = 0. if self.previous_lane is None else (guarded_lane-self.previous_lane)/dt
        if nominal and approved and captured and not conflict:
            geometry = line.sample(progress)
            source = 'variable_nominal_path'
            direct = True
        else:
            geometry = self.offset_geometry(reference, progress, guarded_lane)
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
                reference,line,progress,guarded_lane,source,curvature)
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
        if nominal_pursuit:
            target=np.asarray(line.sample(progress+self.pursuit_lookahead(speed))[0])
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
            steering,_ = self.fallback.control(reference,progress,position,yaw,speed,dt,lateral_offset=guarded_lane)
            unconstrained = steering
            clipped = steering
        self.previous_steer = steering
        self.previous_lane = float(guarded_lane)
        longitudinal_accel=0. if self.previous_speed is None else (speed-self.previous_speed)/dt
        self.previous_speed=float(speed)
        friction=float(self.config.get('friction_coefficient',1.5))*9.81*.95
        available=math.sqrt(max(0.,friction*friction-longitudinal_accel*longitudinal_accel))
        lateral_budget=float(getattr(reference,'lateral_accel',8.))
        corrective_budget=lateral_budget*float(self.config.get('steering_lateral_reserve_factor',1.2))
        allowed_lateral=min(corrective_budget,available)
        speed_steer_limit=min(self.max_steer,math.atan(effective_wheelbase*allowed_lateral/max(speed*speed,25.)))
        before_stability=steering
        if self.kinematic_steering_limit_enabled:
            steering=float(np.clip(steering,-speed_steer_limit,speed_steer_limit))
        self.previous_steer=steering
        return steering, {'version':self.VERSION,'geometry_source':source,
            'steering_method':'variable_path_pure_pursuit' if nominal_pursuit else (
                'curvature_feedforward_feedback' if direct else 'guarded_pure_pursuit_transition'),
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

    def recovery_speed(self, target_speed, metadata):
        factor = float(np.clip(1.-.06*abs(metadata['selected_path_cross_error_m']),.55,1.))
        target = float(target_speed)*factor
        heading_recovery = abs(metadata['selected_path_heading_error_rad']) > .4
        if heading_recovery:
            target = min(target,12.)
        return target, {'path_tracking_speed_factor':factor,'selected_path_heading_recovery':heading_recovery}

    def profile_acceleration(self, reference, line, progress, speed, *, raw_action,
                             actual_lateral, aggressive, current_cap, target_speed, guard_flags=(),tracking_metadata=None):
        if not self.config.get('profile_acceleration_feedforward',True):
            return 0.
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
        return throttle,brake,{'spin_recovery_active':bool(unstable),
            'forward_speed_ratio':ratio,'physical_speed_for_braking_mps':float(body_speed),
            'minimum_stability_decel_mps2':decel}

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
