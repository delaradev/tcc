import tensorflow as tf


def tversky_loss(alpha: float = 0.7, beta: float = 0.3, smooth: float = 1e-6):
    def loss(y_true, y_pred):
        y_true_f = tf.cast(y_true, tf.float32)
        y_pred_f = tf.cast(y_pred, tf.float32)

        tp = tf.reduce_sum(y_true_f * y_pred_f, axis=[1, 2, 3])
        fp = tf.reduce_sum((1 - y_true_f) * y_pred_f, axis=[1, 2, 3])
        fn = tf.reduce_sum(y_true_f * (1 - y_pred_f), axis=[1, 2, 3])

        tversky = (tp + smooth) / (tp + alpha * fp + beta * fn + smooth)
        return 1.0 - tf.reduce_mean(tversky)
    return loss


def dice_loss(smooth: float = 1e-6):
    def loss(y_true, y_pred):
        y_true_f = tf.cast(y_true, tf.float32)
        y_pred_f = tf.cast(y_pred, tf.float32)

        intersection = tf.reduce_sum(y_true_f * y_pred_f, axis=[1, 2, 3])
        union = tf.reduce_sum(y_true_f + y_pred_f, axis=[1, 2, 3])

        dice = (2.0 * intersection + smooth) / (union + smooth)
        return 1.0 - tf.reduce_mean(dice)
    return loss


def combined_loss(alpha: float = 0.5, beta: float = 0.5):
    tversky = tversky_loss()
    bce = tf.keras.losses.BinaryCrossentropy()

    def loss(y_true, y_pred):
        return alpha * tversky(y_true, y_pred) + beta * bce(y_true, y_pred)
    return loss
