from ..get_data import splitSegmentations
import xgboost as xgb

def convertToDMatrix(segmentation):
    dmatrix_train_10p = xgb.DMatrix(segmentation["X_train_10p"], label=segmentation["y_train_10p"], enable_categorical=True)
    dmatrix_train_50p = xgb.DMatrix(segmentation["X_train_50p"], label=segmentation["y_train_50p"], enable_categorical=True)
    dmatrix_train_100p = xgb.DMatrix(segmentation["X_train_100p"], label=segmentation["y_train_100p"], enable_categorical=True)

    dmatrix_test_10p = xgb.DMatrix(segmentation["X_test_10p"], label=segmentation["y_test_10p"], enable_categorical=True)
    dmatrix_test_50p = xgb.DMatrix(segmentation["X_test_50p"], label=segmentation["y_test_50p"], enable_categorical=True)
    dmatrix_test_100p = xgb.DMatrix(segmentation["X_test_100p"], label=segmentation["y_test_100p"], enable_categorical=True)
    
    return {
        "train_10p": dmatrix_train_10p,
        "train_50p": dmatrix_train_50p,
        "train_100p": dmatrix_train_100p,
        "test_10p": dmatrix_test_10p,
        "test_50p": dmatrix_test_50p,
        "test_100p": dmatrix_test_100p,
    }

dmatrixSegmentations = list(map(lambda x: convertToDMatrix(x), splitSegmentations))