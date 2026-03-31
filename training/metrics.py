"""Métricas para segmentação"""
import tensorflow as tf


def iou_score(threshold: float = 0.5, smooth: float = 1e-6):
    """Intersection over Union (Jaccard Index)"""
    def metric(y_true, y_pred):
        y_true = tf.cast(y_true, tf.float32)
        y_pred = tf.cast(y_pred, tf.float32)
        y_pred_bin = tf.where(y_pred >= threshold, 1.0, 0.0)

        intersection = tf.reduce_sum(y_true * y_pred_bin, axis=[1, 2, 3])
        union = tf.reduce_sum(y_true + y_pred_bin,
                              axis=[1, 2, 3]) - intersection

        return tf.reduce_mean((intersection + smooth) / (union + smooth))

    return metric


def dice_score(threshold: float = 0.5, smooth: float = 1e-6):
    """Dice coefficient (F1-score)"""
    def metric(y_true, y_pred):
        y_true = tf.cast(y_true, tf.float32)
        y_pred = tf.cast(y_pred, tf.float32)
        y_pred_bin = tf.where(y_pred >= threshold, 1.0, 0.0)

        intersection = tf.reduce_sum(y_true * y_pred_bin, axis=[1, 2, 3])
        total = tf.reduce_sum(y_true + y_pred_bin, axis=[1, 2, 3])

        return tf.reduce_mean((2.0 * intersection + smooth) / (total + smooth))

    return metric


def precision_score(threshold: float = 0.5, smooth: float = 1e-6):
    """Precision: TP / (TP + FP)"""
    def metric(y_true, y_pred):
        y_true = tf.cast(y_true, tf.float32)
        y_pred = tf.cast(y_pred, tf.float32)
        y_pred_bin = tf.where(y_pred >= threshold, 1.0, 0.0)

        tp = tf.reduce_sum(y_true * y_pred_bin, axis=[1, 2, 3])
        fp = tf.reduce_sum((1 - y_true) * y_pred_bin, axis=[1, 2, 3])

        return tf.reduce_mean((tp + smooth) / (tp + fp + smooth))

    return metric


def recall_score(threshold: float = 0.5, smooth: float = 1e-6):
    """Recall: TP / (TP + FN)"""
    def metric(y_true, y_pred):
        y_true = tf.cast(y_true, tf.float32)
        y_pred = tf.cast(y_pred, tf.float32)
        y_pred_bin = tf.where(y_pred >= threshold, 1.0, 0.0)

        tp = tf.reduce_sum(y_true * y_pred_bin, axis=[1, 2, 3])
        fn = tf.reduce_sum(y_true * (1 - y_pred_bin), axis=[1, 2, 3])

        return tf.reduce_mean((tp + smooth) / (tp + fn + smooth))

    return metric
