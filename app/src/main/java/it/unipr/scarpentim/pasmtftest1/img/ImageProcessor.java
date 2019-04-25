package it.unipr.scarpentim.pasmtftest1.img;

import android.content.Context;
import android.content.res.AssetManager;
import android.graphics.Bitmap;
import android.graphics.BitmapFactory;
import android.util.Log;

import org.opencv.android.Utils;
import org.opencv.core.Core;
import org.opencv.core.Mat;
import org.opencv.core.Point;
import org.opencv.core.Scalar;
import org.opencv.imgcodecs.Imgcodecs;
import org.opencv.imgproc.Imgproc;

import java.io.File;
import java.io.IOException;
import java.io.InputStream;
import java.util.List;

import it.unipr.scarpentim.pasmtftest1.tensorflow.Classifier;

import static org.opencv.core.Core.FILLED;
import static org.opencv.core.Core.LINE_AA;

public class ImageProcessor {

    private static final String TAG = "pasm-YoloClassifier";

    private Mat rgbImage = new Mat();
    private Mat boxesImage = new Mat();
    private Context appContext;

    public ImageProcessor(Context appContext) {
        this.appContext = appContext;
    }

    public void loadImage(Bitmap loadedImage) {
        Bitmap bmp32 = loadedImage.copy(Bitmap.Config.ARGB_8888, true);
        Utils.bitmapToMat(bmp32, rgbImage);
    }

    public Mat drawBoxes(List<Classifier.Recognition> boxes){
        rgbImage.copyTo(boxesImage);
        Scalar green = new Scalar(0, 125, 0);

        for (Classifier.Recognition box : boxes) {
            Log.i(TAG, String.valueOf(box));
            if (box.getConfidence() > 0.3) {
//                green.val[0] = (green.val[0] + 25) % 255;
//                green.val[1] = (green.val[1] + 35) % 255;
//                green.val[2] = (green.val[2] + 45) % 255;
                Point pt1 = new Point(box.getLocation().left, box.getLocation().top);
                Point pt2 = new Point(box.getLocation().right, box.getLocation().bottom);
                Imgproc.rectangle(boxesImage, pt1, pt2, green, 8);
                Point pt3 = new Point(box.getLocation().left, box.getLocation().top);
                Point pt4 = new Point(Math.min(box.getLocation().right, box.getLocation().left + (box.getTitle().length() * 7)), box.getLocation().top + 10);
                Imgproc.rectangle(boxesImage, pt3, pt4, green, FILLED,8);

                pt1.set(new double[] {pt1.x, pt1.y + 8});
                Imgproc.putText(boxesImage, box.getTitle(), pt1, Core.FONT_HERSHEY_DUPLEX, 0.35, new Scalar(255,255,255), 1, LINE_AA);
            }
        }

        return boxesImage;
    }
}
