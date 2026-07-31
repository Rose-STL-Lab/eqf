from datasets.video import (
    DroidVideoDataset,
    MinecraftVideoDataset,
    Re10KVideoDataset,
)
from algorithms import (
    DFoTVideo,
    EqForcingVideo,
    FlowForcingVideo,
    NoiseLevelPredEqFVideo,
    WanForcingVideo,
    Wan22EqFVideo,
    Wan22ForcingVideo,
    Wan22NoiseLevelPredEqFVideo,
    WanT2VPoseForcingVideo,
    WanT2VPoseEqFVideo,
    WanPoseNoiseLevelPredEqFVideo,
)
from .base_exp import BaseLightningExperiment
from .data_modules import _data_module_cls


class VideoGenerationExperiment(BaseLightningExperiment):
    """
    A video generation experiment
    """

    compatible_algorithms = dict(
        dfot_video=DFoTVideo,
        eqf_video=EqForcingVideo,
        flowf_video=FlowForcingVideo,
        noiselevelpred_eqf_video=NoiseLevelPredEqFVideo,
        wan_forcing_video=WanForcingVideo,
        wan22_eqf_video=Wan22EqFVideo,
        wan22_forcing_video=Wan22ForcingVideo,
        wan22_noiselevelpred_eqf_video=Wan22NoiseLevelPredEqFVideo,
        wan_t2v_pose_forcing_video=WanT2VPoseForcingVideo,
        wan_t2v_pose_eqf_video=WanT2VPoseEqFVideo,
        wan_pose_noiselevelpred_eqf_video=WanPoseNoiseLevelPredEqFVideo,
    )

    compatible_datasets = dict(
        droid=DroidVideoDataset,
        minecraft=MinecraftVideoDataset,
        re10k=Re10KVideoDataset,
    )

    data_module_cls = _data_module_cls
