def calculate_mAP(self, classAP_data):
    AP_50_per_class, PR_50_pts_per_class = {}, {}
    (
        num_true_per_class,
        num_positive_per_class,
        num_TP_50_per_class,
        num_FP_50_per_class,
    ) = ({}, {}, {}, {})
    mAP_50, mAP_75, mAP_5095 = 0, 0, 0
    valid_num_classes = 0 + 1e-8

    for res in classAP_data:
        if res["total_positive"] > 0:
            valid_num_classes += 1
            AP_50_per_class[res["class"]] = res["AP_50"]
            PR_50_pts_per_class[res["class"]] = {
                "mprec": res["prec_50"],
                "mrec": res["rec_50"],
            }
            num_true_per_class[res["class"]] = res["total_true"]
            num_positive_per_class[res["class"]] = res["total_positive"]
            num_TP_50_per_class[res["class"]] = res["total_TP_50"]
            num_FP_50_per_class[res["class"]] = res["total_FP_50"]
            mAP_50 += res["AP_50"]
            mAP_75 += res["AP_75"]
            mAP_5095 += res["AP_5095"]

    mAP_50 /= valid_num_classes
    mAP_75 /= valid_num_classes
    mAP_5095 /= valid_num_classes

    res = {
        "AP_50_PER_CLASS": AP_50_per_class,
        "PR_50_PTS_PER_CLASS": PR_50_pts_per_class,
        "NUM_TRUE_PER_CLASS": num_true_per_class,
        "NUM_POSITIVE_PER_CLASS": num_positive_per_class,
        "NUM_TP_50_PER_CLASS": num_TP_50_per_class,
        "NUM_FP_50_PER_CLASS": num_FP_50_per_class,
        "mAP_50": round(mAP_50, 4),
        "mAP_75": round(mAP_75, 4),
        "mAP_5095": round(mAP_5095, 4),
    }
    return res
