
python tools/result_converter/det_result_late_fusion.py \
    --config projects/configs_griffin_50scenes_25m/cooperative/tiny_track_r50_stream_bs1_3cls_late_fusion.py \
    --veh-det-pkl projects/work_dirs_griffin_50scenes_25m/vehicle-side/tiny_track_r50_stream_bs8_48epoch_3cls/results-02211032.pkl \
    --inf-det-pkl projects/work_dirs_griffin_50scenes_25m/drone-side/tiny_track_r50_stream_bs8_48epoch_3cls/results-02211258.pkl \
    --out projects/work_dirs_griffin_50scenes_25m/cooperative/tiny_track_r50_stream_bs1_3cls_late_fusion/results.pkl \
    # --debug

python tools/result_converter/det_result_late_fusion.py \
    --config projects/configs_griffin_50scenes_25m/cooperative/tiny_track_r50_stream_bs1_3cls_late_fusion_200latency.py \
    --veh-det-pkl projects/work_dirs_griffin_50scenes_25m/vehicle-side/tiny_track_r50_stream_bs8_48epoch_3cls/results-02211032.pkl \
    --inf-det-pkl projects/work_dirs_griffin_50scenes_25m/drone-side/tiny_track_r50_stream_bs8_48epoch_3cls/results-02211258.pkl \
    --out projects/work_dirs_griffin_50scenes_25m/cooperative/tiny_track_r50_stream_bs1_3cls_late_fusion_200latency/results.pkl \
    # --debug

python tools/result_converter/det_result_late_fusion.py \
    --config projects/configs_griffin_50scenes_25m/cooperative/tiny_track_r50_stream_bs1_3cls_late_fusion_400latency.py \
    --veh-det-pkl projects/work_dirs_griffin_50scenes_25m/vehicle-side/tiny_track_r50_stream_bs8_48epoch_3cls/results-02211032.pkl \
    --inf-det-pkl projects/work_dirs_griffin_50scenes_25m/drone-side/tiny_track_r50_stream_bs8_48epoch_3cls/results-02211258.pkl \
    --out projects/work_dirs_griffin_50scenes_25m/cooperative/tiny_track_r50_stream_bs1_3cls_late_fusion_400latency/results.pkl \
    # --debug

python tools/result_converter/det_result_late_fusion.py \
    --config projects/configs_griffin_50scenes_40m/cooperative/tiny_track_r50_stream_bs1_3cls_late_fusion.py \
    --veh-det-pkl projects/work_dirs_griffin_50scenes_40m/vehicle-side/tiny_track_r50_stream_bs8_48epoch_3cls/results-02242239.pkl \
    --inf-det-pkl projects/work_dirs_griffin_50scenes_40m/drone-side/tiny_track_r50_stream_bs8_24epoch_3cls_eval/results-02251541.pkl \
    --out projects/work_dirs_griffin_50scenes_40m/cooperative/tiny_track_r50_stream_bs1_3cls_late_fusion/results.pkl \
    # --debug

python tools/result_converter/det_result_late_fusion.py \
    --config projects/configs_griffin_100scenes_random/cooperative/tiny_track_r50_stream_bs1_3cls_late_fusion.py \
    --veh-det-pkl projects/work_dirs_griffin_100scenes_random/vehicle-side/tiny_track_r50_stream_bs8_48epoch_3cls/results-02242111.pkl \
    --inf-det-pkl projects/work_dirs_griffin_100scenes_random/drone-side/tiny_track_r50_stream_bs8_24epoch_3cls_eval/results-02242105.pkl \
    --out projects/work_dirs_griffin_100scenes_random/cooperative/tiny_track_r50_stream_bs1_3cls_late_fusion/results.pkl \
    # --debug
