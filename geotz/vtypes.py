"""Vehicle classes shared by both sites.

HV  human-driven vehicle (IDM, human reaction gap)
CHV connected human-driven vehicle (IDM, shorter gap when following a connected leader, cooperative)
AV  automated, not connected (ACC)
CAV connected automated vehicle (CACC towards connected leaders, ACC otherwise)
TRK heavy vehicle, human driven (always HV behaviour)

Platoon logic (run time, see simcore.PlatoonManager): a CAV whose leader is a CAV within a
short gap joins that leader's string; a CAV behind a CHV forms a mixed connected platoon with
a longer gap. When the string reaches the maximum platoon size the next CAV keeps the ACC gap,
which opens an inter-platoon gap.
"""

TAU = dict(HV=1.5, CHV_free=1.5, CHV_conn=1.1, AV=1.2, CAV_acc=1.2, CAV_cacc=0.6, CAV_mixed=0.9)

VTYPES_TMPL = """
  <vType id="HV" vClass="passenger" carFollowModel="IDM" accel="1.6" decel="3.0" emergencyDecel="7.5" tau="{HV}" minGap="2.0" delta="4" length="4.8" speedFactor="normc(1,0.1,0.7,1.3)" lcStrategic="{hv_strat}" lcCooperative="0.6" lcSpeedGain="{hv_sg}" lcKeepRight="0.5" lcAssertive="{hv_assert}" color="0.6,0.6,0.6"/>
  <vType id="CHV" vClass="passenger" carFollowModel="IDM" accel="1.6" decel="3.0" emergencyDecel="7.5" tau="{CHV_free}" minGap="2.0" delta="4" length="4.8" speedFactor="normc(1,0.1,0.7,1.3)" lcStrategic="{hv_strat}" lcCooperative="1.0" lcSpeedGain="{hv_sg}" lcKeepRight="0.5" lcAssertive="{hv_assert}" color="0.2,0.6,1"/>
  <vType id="AV" vClass="passenger" carFollowModel="ACC" accel="1.5" decel="3.0" emergencyDecel="7.5" tau="{AV}" minGap="2.0" length="4.8" speedFactor="normc(1,0.02,0.95,1.05)" lcStrategic="1.0" lcCooperative="0.8" lcSpeedGain="0.8" lcKeepRight="0.5" color="0,0.8,0"/>
  <vType id="CAV" vClass="custom1" carFollowModel="CACC" accel="1.5" decel="3.0" emergencyDecel="7.5" tau="{CAV_cacc}" minGap="2.0" length="4.8" speedFactor="normc(1,0.02,0.95,1.05)" lcStrategic="1.0" lcCooperative="1.0" lcSpeedGain="0.8" lcKeepRight="0.0" color="1,0,0"/>
  <vType id="TRK" vClass="truck" carFollowModel="IDM" accel="0.8" decel="2.5" emergencyDecel="6.0" tau="1.8" minGap="2.5" delta="4" length="12.0" speedFactor="normc(0.9,0.05,0.7,1.1)" lcStrategic="1.0" lcCooperative="0.6" lcSpeedGain="0.5" lcKeepRight="1.0" color="0.4,0.2,0"/>
"""

HV_LC = dict(hv_strat=1.0, hv_sg=1.0, hv_assert=1.0)   # replaced by the calibrated values (calib_b.json)


def vtypes_xml(hv_lc=None):
    p = dict(HV_LC); p.update(hv_lc or {})
    tau = dict(TAU)
    if 'HV' in p:
        tau['HV'] = p.pop('HV'); tau['CHV_free'] = tau['HV']
    return VTYPES_TMPL.format(**tau, **p)

CAR_CLASSES = ['HV', 'CHV', 'AV', 'CAV']


def class_shares(p_cav, mix=(0.7, 0.2, 0.1)):
    """Shares of the passenger-car fleet. The non-CAV remainder is split HV:CHV:AV = mix."""
    r = 1.0 - p_cav
    return dict(HV=r * mix[0], CHV=r * mix[1], AV=r * mix[2], CAV=p_cav)
