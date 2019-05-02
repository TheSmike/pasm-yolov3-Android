package it.unipr.scarpentim.pasmtftest1.img;

import android.os.Environment;

import org.opencv.core.Mat;
import org.opencv.imgcodecs.Imgcodecs;
import org.opencv.imgproc.Imgproc;

import java.io.File;
import java.text.DateFormat;
import java.text.SimpleDateFormat;
import java.util.Calendar;
import java.util.Date;

public class ImageSaver {

    DateFormat df = new SimpleDateFormat("yyyyMMdd_HHmmss");
    public static final String FOLDER_NAME = "pasmApp";
    File dir = null;
    Date previousTime = null;
    int previousProg = 0;

    public void createFolderIfNotExist() {
        dir = new File(Environment.getExternalStoragePublicDirectory(Environment.DIRECTORY_DCIM), FOLDER_NAME);
        if (!dir.exists()) {
            dir.mkdirs();
        }
    }

    public boolean save(Mat mat) {
        Mat ultimate = new Mat();
        Imgproc.cvtColor(mat, ultimate, Imgproc.COLOR_RGB2BGR);

        Date currentTime = Calendar.getInstance().getTime();
        String strProg = "";
        if (currentTime.equals(previousTime)) {
            strProg = String.valueOf(++previousProg);
        }else{
            previousProg = 0;
        }
        String fullPath = String.format("%s/recognition_%s%s.jpg", dir.getAbsolutePath(), df.format(currentTime), strProg);
        previousTime = currentTime;
        return Imgcodecs.imwrite(fullPath, ultimate);
    }
}
