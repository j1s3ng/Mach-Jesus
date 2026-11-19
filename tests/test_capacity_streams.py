from scripts.audit_capacity_streams import stream_summary


def records():
    return [{'timestamp_ns':1_000_000_000+200_000_000*i,'frame_id':str(i),'points':15328,'inference_ms':40.}for i in range(5)]


def test_full_stream_has_requested_simulated_cadence():
    result=stream_summary(records(),start_s=1.,duration_s=1.,period_ms=200)
    assert result['verified'] and result['observed_frames']==5


def test_early_actor_termination_does_not_become_smaller_successful_field():
    result=stream_summary(records()[:3],start_s=1.,duration_s=1.,period_ms=200)
    assert not result['verified'] and 'early_stream_termination'in result['issues']


def test_dropped_or_duplicate_frames_and_invalid_latency_fail():
    values=records();values[2]=dict(values[1]);values[4]['inference_ms']=float('nan')
    result=stream_summary(values,start_s=1.,duration_s=1.,period_ms=200)
    assert not result['verified']
    assert {'duplicate_frame_id','cadence_gap_or_nonmonotonic_clock','invalid_RPC_clock'}<=set(result['issues'])
