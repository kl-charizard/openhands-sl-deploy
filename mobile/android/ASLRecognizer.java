package com.example.aslrecognizer;

import android.content.Context;
import android.graphics.Bitmap;
import android.util.Log;

import org.tensorflow.lite.Interpreter;
import org.tensorflow.lite.support.common.FileUtil;
import org.tensorflow.lite.support.image.TensorImage;
import org.tensorflow.lite.support.tensorbuffer.TensorBuffer;

import java.io.IOException;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.util.ArrayList;
import java.util.Collections;
import java.util.List;

/**
 * 🤖 OpenHands ASL Recognition for Android
 * TensorFlow Lite implementation with pose-based inference
 */
public class ASLRecognizer {
    private static final String TAG = "ASLRecognizer";
    
    // Model configuration
    private static final String MODEL_PATH = "asl_model_int8.tflite";
    private static final int INPUT_SIZE = 158;  // Pose features (hand + body)
    private static final int OUTPUT_SIZE = 2000; // ASL vocabulary
    private static final float CONFIDENCE_THRESHOLD = 0.7f;
    
    // TensorFlow Lite components
    private Interpreter interpreter;
    private String[] labels;
    private Context context;
    
    // Pose extractor
    private PoseExtractor poseExtractor;
    
    public ASLRecognizer(Context context) throws IOException {
        this.context = context;
        initializeModel();
        initializePoseExtractor();
        loadLabels();
    }
    
    /**
     * Initialize TensorFlow Lite model
     */
    private void initializeModel() throws IOException {
        Log.d(TAG, "🚀 Initializing OpenHands ASL model...");
        
        try {
            // Load quantized model
            ByteBuffer modelBuffer = FileUtil.loadMappedFile(context, MODEL_PATH);
            
            // Configure interpreter options
            Interpreter.Options options = new Interpreter.Options();
            options.setNumThreads(4); // Optimize for mobile CPU
            options.setUseNNAPI(true); // Use Android Neural Networks API
            
            // Create interpreter
            interpreter = new Interpreter(modelBuffer, options);
            
            Log.d(TAG, "✅ Model loaded successfully");
            Log.d(TAG, "📊 Input shape: " + java.util.Arrays.toString(interpreter.getInputTensor(0).shape()));
            Log.d(TAG, "📊 Output shape: " + java.util.Arrays.toString(interpreter.getOutputTensor(0).shape()));
            
        } catch (IOException e) {
            Log.e(TAG, "❌ Failed to load model: " + e.getMessage());
            throw e;
        }
    }
    
    /**
     * Initialize pose extractor (MediaPipe equivalent)
     */
    private void initializePoseExtractor() {
        Log.d(TAG, "🖐️ Initializing pose extractor...");
        poseExtractor = new PoseExtractor(context);
        Log.d(TAG, "✅ Pose extractor ready");
    }
    
    /**
     * Load ASL vocabulary labels
     */
    private void loadLabels() throws IOException {
        Log.d(TAG, "📚 Loading ASL vocabulary...");
        
        try {
            List<String> labelList = FileUtil.loadLabels(context, "asl_labels.txt");
            labels = labelList.toArray(new String[0]);
            
            Log.d(TAG, "✅ Loaded " + labels.length + " ASL signs");
            
        } catch (IOException e) {
            Log.e(TAG, "❌ Failed to load labels: " + e.getMessage());
            // Create dummy labels as fallback
            labels = new String[OUTPUT_SIZE];
            for (int i = 0; i < OUTPUT_SIZE; i++) {
                labels[i] = "SIGN_" + i;
            }
            Log.w(TAG, "⚠️ Using dummy labels");
        }
    }
    
    /**
     * Recognize ASL sign from camera frame
     * 
     * @param bitmap Camera frame as bitmap
     * @return Recognition result
     */
    public ASLResult recognizeSign(Bitmap bitmap) {
        long startTime = System.currentTimeMillis();
        
        try {
            // Extract pose features
            float[] poseFeatures = poseExtractor.extractPoseFeatures(bitmap);
            
            if (poseFeatures == null) {
                return new ASLResult("", 0.0f, "No hands detected", 
                                   System.currentTimeMillis() - startTime);
            }
            
            // Prepare input tensor
            ByteBuffer inputBuffer = ByteBuffer.allocateDirect(INPUT_SIZE * 4);
            inputBuffer.order(ByteOrder.nativeOrder());
            
            for (float feature : poseFeatures) {
                inputBuffer.putFloat(feature);
            }
            inputBuffer.rewind();
            
            // Prepare output tensor
            ByteBuffer outputBuffer = ByteBuffer.allocateDirect(OUTPUT_SIZE * 4);
            outputBuffer.order(ByteOrder.nativeOrder());
            
            // Run inference
            interpreter.run(inputBuffer, outputBuffer);
            
            // Process output
            outputBuffer.rewind();
            float[] probabilities = new float[OUTPUT_SIZE];
            for (int i = 0; i < OUTPUT_SIZE; i++) {
                probabilities[i] = outputBuffer.getFloat();
            }
            
            // Find best prediction
            int bestIndex = 0;
            float bestConfidence = probabilities[0];
            
            for (int i = 1; i < probabilities.length; i++) {
                if (probabilities[i] > bestConfidence) {
                    bestConfidence = probabilities[i];
                    bestIndex = i;
                }
            }
            
            // Create result
            String predictedSign = bestIndex < labels.length ? labels[bestIndex] : "UNKNOWN";
            String message = bestConfidence > CONFIDENCE_THRESHOLD ? 
                           "Recognized: " + predictedSign : 
                           "Low confidence: " + String.format("%.2f", bestConfidence);
            
            return new ASLResult(
                bestConfidence > CONFIDENCE_THRESHOLD ? predictedSign : "",
                bestConfidence,
                message,
                System.currentTimeMillis() - startTime
            );
            
        } catch (Exception e) {
            Log.e(TAG, "❌ Recognition error: " + e.getMessage());
            return new ASLResult("", 0.0f, "Error: " + e.getMessage(),
                               System.currentTimeMillis() - startTime);
        }
    }
    
    /**
     * Get top N predictions
     */
    public List<ASLResult> getTopPredictions(Bitmap bitmap, int topN) {
        // Implementation similar to recognizeSign but returns multiple results
        List<ASLResult> results = new ArrayList<>();
        // ... (implementation details)
        return results;
    }
    
    /**
     * Build sentence from sequence of recognitions
     */
    public String buildSentence(List<ASLResult> recognitionHistory) {
        if (recognitionHistory.isEmpty()) {
            return "";
        }
        
        List<String> sentenceParts = new ArrayList<>();
        String currentSign = null;
        
        // Group consecutive similar signs
        for (ASLResult result : recognitionHistory) {
            if (result.getConfidence() > CONFIDENCE_THRESHOLD && 
                !result.getPredictedSign().isEmpty() &&
                !result.getPredictedSign().equals(currentSign)) {
                
                sentenceParts.add(result.getPredictedSign());
                currentSign = result.getPredictedSign();
            }
        }
        
        return String.join(" ", sentenceParts);
    }
    
    /**
     * Get model performance info
     */
    public ModelInfo getModelInfo() {
        return new ModelInfo(
            MODEL_PATH,
            INPUT_SIZE,
            OUTPUT_SIZE,
            labels.length,
            "INT8 Quantized",
            interpreter != null ? "Ready" : "Not loaded"
        );
    }
    
    /**
     * Cleanup resources
     */
    public void close() {
        if (interpreter != null) {
            interpreter.close();
            interpreter = null;
        }
        
        if (poseExtractor != null) {
            poseExtractor.close();
            poseExtractor = null;
        }
        
        Log.d(TAG, "✅ ASLRecognizer resources cleaned up");
    }
    
    /**
     * ASL Recognition Result
     */
    public static class ASLResult {
        private final String predictedSign;
        private final float confidence;
        private final String message;
        private final long inferenceTimeMs;
        
        public ASLResult(String predictedSign, float confidence, String message, long inferenceTimeMs) {
            this.predictedSign = predictedSign;
            this.confidence = confidence;
            this.message = message;
            this.inferenceTimeMs = inferenceTimeMs;
        }
        
        // Getters
        public String getPredictedSign() { return predictedSign; }
        public float getConfidence() { return confidence; }
        public String getMessage() { return message; }
        public long getInferenceTimeMs() { return inferenceTimeMs; }
        
        @Override
        public String toString() {
            return String.format("ASLResult{sign='%s', confidence=%.3f, time=%dms}", 
                               predictedSign, confidence, inferenceTimeMs);
        }
    }
    
    /**
     * Model Information
     */
    public static class ModelInfo {
        private final String modelPath;
        private final int inputSize;
        private final int outputSize;
        private final int vocabularySize;
        private final String quantization;
        private final String status;
        
        public ModelInfo(String modelPath, int inputSize, int outputSize, 
                        int vocabularySize, String quantization, String status) {
            this.modelPath = modelPath;
            this.inputSize = inputSize;
            this.outputSize = outputSize;
            this.vocabularySize = vocabularySize;
            this.quantization = quantization;
            this.status = status;
        }
        
        // Getters
        public String getModelPath() { return modelPath; }
        public int getInputSize() { return inputSize; }
        public int getOutputSize() { return outputSize; }
        public int getVocabularySize() { return vocabularySize; }
        public String getQuantization() { return quantization; }
        public String getStatus() { return status; }
    }
}
