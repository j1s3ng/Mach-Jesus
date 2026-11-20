import math
import metric as m

def box(x=0.,score=.9):return {'center':[x,0,0],'size':[4,2,2],'yaw':0.,'score':score}
def frame(boxes):return {'scene_id':'s','frame_id':'f','boxes':boxes}
def truth(eligible,ignored=()):return {('s','f'):{'eligible':[m.polygon(x) for x in eligible],'ignored':[m.polygon(x) for x in ignored]}}

def test_perfect_prediction():
 r=m.score(truth([box()]),[frame([box()])]);assert r['ap101']==1 and r['tp']==1 and r['fp']==0

def test_ranked_false_positive_changes_precision():
 r=m.score(truth([box()]),[frame([box(20,.99),box(0,.8)])]);assert r['ap101']==.5 and r['tp']==1 and r['fp']==1

def test_duplicate_is_false_positive_and_unmatched_target_is_fn():
 r=m.score(truth([box(0),box(20)]),[frame([box(0,.9),box(0,.8)])]);assert r['tp']==1 and r['fp']==1 and math.isclose(r['ap101'],51/101)

def test_excluded_region_ignored_and_eligible_has_precedence():
 r=m.score(truth([box(0)],[box(0),box(20)]),[frame([box(20,.99),box(0,.9)])]);assert r['ap101']==1 and r['tp']==1 and r['fp']==0 and r['ignored_predictions']==1

def test_empty_frame_and_floor():
 t=truth([box()]);t[('s','empty')]={'eligible':[],'ignored':[]}
 p=[frame([box(0,.8),box(30,.09)]),{'scene_id':'s','frame_id':'empty','boxes':[box(20,.9)]}]
 r=m.score(t,p);assert r['ap101']==.5 and r['fp']==1 and r['scored_predictions']==2
