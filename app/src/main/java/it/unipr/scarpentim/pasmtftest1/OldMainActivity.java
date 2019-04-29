package it.unipr.scarpentim.pasmtftest1;

import android.Manifest;
import android.content.Intent;
import android.content.pm.PackageManager;
import android.database.Cursor;
import android.graphics.Bitmap;
import android.graphics.BitmapFactory;
import android.net.Uri;
import android.os.AsyncTask;
import android.provider.MediaStore;
import android.support.v4.app.ActivityCompat;
import android.support.v4.content.ContextCompat;
import android.support.v7.app.AppCompatActivity;
import android.os.Bundle;
import android.util.Log;
import android.view.Menu;
import android.view.MenuItem;
import android.view.SurfaceView;
import android.view.View;
import android.view.WindowManager;
import android.widget.ImageView;
import android.widget.TextView;

import org.opencv.android.BaseLoaderCallback;
import org.opencv.android.CameraBridgeViewBase;
import org.opencv.android.LoaderCallbackInterface;
import org.opencv.android.OpenCVLoader;
import org.opencv.android.Utils;
import org.opencv.core.Core;
import org.opencv.core.CvType;
import org.opencv.core.Mat;
import org.opencv.core.Size;
import org.opencv.imgproc.Imgproc;

import java.io.IOException;
import java.util.List;

import it.unipr.scarpentim.pasmtftest1.tensorflow.Classifier;
import it.unipr.scarpentim.pasmtftest1.tensorflow.TensorFlowImageClassifier;

public class OldMainActivity extends AppCompatActivity implements CameraBridgeViewBase.CvCameraViewListener2 {


    private static final int MY_PERMISSIONS_REQUEST_CAMERA = 1;
    private static final int MY_PERMISSIONS_REQUEST_STORAGE = 1;

    private boolean disable = false;
    private Classifier classifier = null;
    private CameraBridgeViewBase mOpenCvCameraView;

    Mat mRgba;
    Mat mRgbaF;
    Mat mRgbaT;

    private static final String TAG = "PASM_yolov3";
    private static final int SELECT_PICTURE = 1;
    private String selectedImagePath;
    private Bitmap myBitmap = null;

    private Menu mOptionsMenu;


    private static final int INPUT_WIDTH = 300;
    private static final int INPUT_HEIGHT = 300;
    private static final int IMAGE_MEAN = 128;
    private static final float IMAGE_STD = 128f;
    private static final String INPUT_NAME = "Mul";
    private static final String OUTPUT_NAME = "final_result";
    private static final String MODEL_FILE = "file:///android_asset/hero_stripped_graph.pb";
    private static final String LABEL_FILE = "file:///android_asset/hero_labels.txt";



    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        getWindow().addFlags(WindowManager.LayoutParams.FLAG_KEEP_SCREEN_ON);
        getWindow().addFlags(WindowManager.LayoutParams.FLAG_KEEP_SCREEN_ON);

        setContentView(R.layout.activity_main);
        validateCameraPermission();
        initClassifier();
        mOpenCvCameraView= findViewById(R.id.cameraView1);
        mOpenCvCameraView.setVisibility(SurfaceView.VISIBLE);
//        mOpenCvCameraView.setMaxFrameSize(1280, 720);
        mOpenCvCameraView.setCvCameraViewListener(this);

        //Log.i(TAG, "mOpenCvCameraView size (w,h):" + mOpenCvCameraView.getWidth() + mOpenCvCameraView.getHeight());

    }


    @Override
    public boolean onCreateOptionsMenu(Menu menu) {
        getMenuInflater().inflate(R.menu.mymenu, menu);
        mOptionsMenu = menu;
        return true;
    }

    @Override
    public boolean onOptionsItemSelected(MenuItem item) {
        int id = item.getItemId();
        if (id == R.id.action_openGallery) {
            validateReadStoragePermission();
            Intent intent = new Intent();
            intent.setType("image/*");
            intent.setAction(Intent.ACTION_PICK);
            startActivityForResult(Intent.createChooser(intent,"Seleziona immagine"), SELECT_PICTURE);
            return true;
        }else if (id == R.id.action_openCamera) {
            ImageView iv = findViewById(R.id.ivGallery);
            iv.setVisibility(View.GONE);
            myBitmap = null;
            mOpenCvCameraView.enableView();
            mOptionsMenu.getItem(1).setEnabled(false);
        }
        return super.onOptionsItemSelected(item);
    }

    Bitmap smallBitmap = null;

    public void classify(View view) {

        Mat mRgbaTemp = mRgba.clone();
        //mOpenCvCameraView.disableView();
        //mOpenCvCameraView.setVisibility(View.GONE);
        new ComputeTask().execute(mRgbaTemp);
        TextView tv = findViewById(R.id.textView);
        tv.setText("I'm thinking...");
    }

    //region - Metodi da CvCameraViewListener2

    @Override
    public void onCameraViewStarted(int width, int height) {
        mRgba = new Mat(height, width, CvType.CV_8UC4);
        mRgbaF = new Mat(width, height, CvType.CV_8UC4);
        mRgbaT = new Mat(width, height, CvType.CV_8UC4);
        Log.i(TAG, "height : " + height);
        Log.i(TAG, "width : " + width);
        //Log.i(TAG, "mOpenCvCameraView size (w,h):" + mOpenCvCameraView.getWidth() +  " - " + mOpenCvCameraView.getHeight());
    }

    @Override
    public void onCameraViewStopped() {
        mRgba.release();
        mRgbaF.release();
        mRgbaT.release();
    }

    @Override
    public Mat onCameraFrame(CameraBridgeViewBase.CvCameraViewFrame inputFrame) {
        mRgba = inputFrame.rgba();
        return inputFrame.rgba();
    }
    //endregion

    @Override
    protected void onResume() {
        super.onResume();
        // Chiama l'inizializzazione asincrona e passa l'oggetto callback
        // creato in precendeza, e sceglie quale versione di OpenCV caricare.
        // Serve anche a verificare che l'OpenCV manager installato supporti
        // la versione che si sta provando a caricare.
        if (!disable)
            OpenCVLoader.initAsync(OpenCVLoader.OPENCV_VERSION,this, mLoaderCallback);
    }

    private class ComputeTask extends AsyncTask<Mat, Bitmap, List<Classifier.Recognition>>{

        @Override
        protected List<Classifier.Recognition> doInBackground(Mat... mats) {
            Mat mRgbaTemp = mats[0];
            if (myBitmap != null){
                smallBitmap = Bitmap.createScaledBitmap(myBitmap, INPUT_WIDTH, INPUT_HEIGHT, false);
            }else{

                smallBitmap = Bitmap.createBitmap(INPUT_WIDTH, INPUT_HEIGHT, Bitmap.Config.RGB_565);
                Bitmap bigBitmap = Bitmap.createBitmap(mRgbaF.width(), mRgbaF.height(), Bitmap.Config.RGB_565);
                Mat mRgbaFixedSize = new Mat(INPUT_WIDTH, INPUT_HEIGHT, CvType.CV_8UC4);

                Core.transpose(mRgbaTemp, mRgbaT);
                Imgproc.resize(mRgbaT, mRgbaF, mRgbaF.size(), 0,0, 0);
                Core.flip(mRgbaF, mRgbaTemp, 1 );

                Imgproc.resize(mRgbaTemp, mRgbaFixedSize, new Size(INPUT_WIDTH, INPUT_HEIGHT), 0,0, 0);

                Utils.matToBitmap(mRgbaFixedSize, smallBitmap);
                Utils.matToBitmap(mRgbaTemp, bigBitmap);

                this.publishProgress(bigBitmap);


                //OLD Toast.makeText(getApplicationContext(), "Nessuna immagine caricata", Toast.LENGTH_SHORT).show();
            }

            List<Classifier.Recognition> recognitions = classifier.recognizeImage(smallBitmap);
            return  recognitions;
        }

        @Override
        protected void onPostExecute(List<Classifier.Recognition> result) {
            TextView tv = findViewById(R.id.textView);
            Log.i(TAG, String.valueOf(result));
            for (Classifier.Recognition recognition : result) {
                Log.i(TAG, "results: " + recognition.getTitle() + " " + recognition.getConfidence());
            }
            if (result.size() > 0)
                tv.setText("This is.... " + result.get(0).getTitle());
            else
                tv.setText("This is.... Not a super hero");

            ImageView iv = findViewById(R.id.imageView1);
            iv.setImageBitmap(smallBitmap);
//            mOpenCvCameraView.enableView();
//            mOpenCvCameraView.setVisibility(View.VISIBLE);


        }

        @Override
        protected void onProgressUpdate(Bitmap... values) {
            Log.i(TAG, "### onProgressUpdate called!!");
            super.onProgressUpdate(values[0]);
            ImageView iv = findViewById(R.id.imageView1);
            iv.setImageBitmap(values[0]);
            iv.setVisibility(View.VISIBLE);
        }
    }

    private void validateReadStoragePermission() {
        if (ContextCompat.checkSelfPermission(OldMainActivity.this, Manifest.permission.READ_EXTERNAL_STORAGE) != PackageManager.PERMISSION_GRANTED) {
            Log.i(TAG, "Permission is not granted");
            ActivityCompat.requestPermissions(OldMainActivity.this,
                    new String[]{Manifest.permission.READ_EXTERNAL_STORAGE},
                    MY_PERMISSIONS_REQUEST_STORAGE);
        }
    }

    private void validateCameraPermission() {
        if (ContextCompat.checkSelfPermission(OldMainActivity.this, Manifest.permission.CAMERA) != PackageManager.PERMISSION_GRANTED) {
            Log.i(TAG, "Permission for camera is not granted");
            ActivityCompat.requestPermissions(OldMainActivity.this,
                    new String[]{Manifest.permission.CAMERA},
                    MY_PERMISSIONS_REQUEST_CAMERA);
        }
    }

    @Override
    public void onActivityResult(int requestCode, int resultCode, Intent data) {
        if (resultCode == RESULT_OK) {
            if (requestCode == SELECT_PICTURE) {
                mOptionsMenu.getItem(1).setEnabled(true);
                disable = true;
                mOpenCvCameraView.disableView();
                Uri selectedImageUri = data.getData();
                selectedImagePath = getPath(selectedImageUri);
                Log.i(TAG, "selectedImagePath: " + selectedImagePath);
                //loadImage(selectedImagePath);
                myBitmap = BitmapFactory.decodeFile(selectedImagePath);
                ImageView iv = findViewById(R.id.ivGallery);
                iv.setImageBitmap(myBitmap);
            }
        }
    }

    private String getPath(Uri uri) {
        if (uri == null) {
            return null;
        }
        // prova a recuperare l'immagine prima dal Media Store
        // questo però funziona solo per immagini selezionate dalla galleria
        String[] projection = {MediaStore.Images.Media.DATA};
        Cursor cursor = getContentResolver().query(uri, projection,null, null, null);
        if (cursor != null) {
            int column_index = cursor.getColumnIndexOrThrow(MediaStore.Images.Media.DATA);
            cursor.moveToFirst();
            return cursor.getString(column_index);
        }
        return uri.getPath();
    }

    private void initClassifier() {
        try {
            classifier = TensorFlowImageClassifier.create(
                    super.getAssets(), MODEL_FILE, LABEL_FILE, INPUT_WIDTH, INPUT_HEIGHT,
                    IMAGE_MEAN, IMAGE_STD, INPUT_NAME, OUTPUT_NAME);
        } catch (IOException e) {
            throw new RuntimeException("classifier init problem", e);
        }
    }

    // Questo oggetto callback è usato quando inizializzaimo la libreria OpenCV in modo asincrono
    private BaseLoaderCallback mLoaderCallback = new BaseLoaderCallback(this) {
        @Override
        // Una volta che OpenCV manager è connesso viene chiamato questo metodo di
        public void onManagerConnected(int status) {
            switch (status) {
                // Una volta che OpenCV manager si è connesso con successo
                // possiamo abilitare l'interazione con la tlc
                case LoaderCallbackInterface.SUCCESS:
                    Log.i(TAG, "OpenCV loaded successfully");
                    mOpenCvCameraView.enableView();

                    break;
                default:
                    super.onManagerConnected(status);
                     break;
            }
        }
    };
}
