import Foundation
import CoreML
import Vision
import UIKit
import AVFoundation

/**
 * 🤖 OpenHands ASL Recognition for iOS
 * Core ML implementation with pose-based inference
 */
@available(iOS 13.0, *)
class ASLRecognizer: ObservableObject {
    
    // MARK: - Properties
    
    private var model: ASLClassifier?
    private var poseExtractor: PoseExtractor
    private let confidenceThreshold: Float = 0.7
    private let inputSize = 158  // Pose features
    private let outputSize = 2000  // ASL vocabulary
    
    @Published var isReady = false
    @Published var recognitionHistory: [ASLResult] = []
    
    // MARK: - Initialization
    
    init() {
        self.poseExtractor = PoseExtractor()
        setupModel()
    }
    
    /**
     * Setup Core ML model
     */
    private func setupModel() {
        DispatchQueue.global(qos: .userInitiated).async { [weak self] in
            do {
                print("🚀 Loading OpenHands ASL model...")
                
                // Load Core ML model
                let configuration = MLModelConfiguration()
                configuration.computeUnits = .all  // Use GPU if available
                
                self?.model = try ASLClassifier(configuration: configuration)
                
                DispatchQueue.main.async {
                    self?.isReady = true
                    print("✅ ASL model loaded successfully")
                }
                
            } catch {
                print("❌ Failed to load ASL model: \(error)")
                DispatchQueue.main.async {
                    self?.isReady = false
                }
            }
        }
    }
    
    // MARK: - Recognition Methods
    
    /**
     * Recognize ASL sign from camera frame
     */
    func recognizeSign(from pixelBuffer: CVPixelBuffer) -> ASLResult {
        let startTime = CFAbsoluteTimeGetCurrent()
        
        guard isReady, let model = model else {
            return ASLResult(
                predictedSign: "",
                confidence: 0.0,
                message: "Model not ready",
                inferenceTimeMs: 0
            )
        }
        
        do {
            // Extract pose features
            guard let poseFeatures = poseExtractor.extractPoseFeatures(from: pixelBuffer) else {
                let inferenceTime = Int64((CFAbsoluteTimeGetCurrent() - startTime) * 1000)
                return ASLResult(
                    predictedSign: "",
                    confidence: 0.0,
                    message: "No hands detected",
                    inferenceTimeMs: inferenceTime
                )
            }
            
            // Convert to MLMultiArray
            let inputArray = try MLMultiArray(shape: [1, NSNumber(value: inputSize)], dataType: .float32)
            for (index, feature) in poseFeatures.enumerated() {
                if index < inputSize {
                    inputArray[index] = NSNumber(value: feature)
                }
            }
            
            // Run prediction
            let prediction = try model.prediction(input: inputArray)
            
            // Process output
            let probabilities = prediction.output
            var bestIndex = 0
            var bestConfidence: Float = 0.0
            
            // Find highest confidence prediction
            for i in 0..<min(outputSize, probabilities.count) {
                let confidence = probabilities[i].floatValue
                if confidence > bestConfidence {
                    bestConfidence = confidence
                    bestIndex = i
                }
            }
            
            // Create result
            let inferenceTime = Int64((CFAbsoluteTimeGetCurrent() - startTime) * 1000)
            let predictedSign = getSignLabel(for: bestIndex)
            let message = bestConfidence > confidenceThreshold ?
                         "Recognized: \(predictedSign)" :
                         "Low confidence: \(String(format: "%.2f", bestConfidence))"
            
            return ASLResult(
                predictedSign: bestConfidence > confidenceThreshold ? predictedSign : "",
                confidence: bestConfidence,
                message: message,
                inferenceTimeMs: inferenceTime
            )
            
        } catch {
            print("❌ Recognition error: \(error)")
            let inferenceTime = Int64((CFAbsoluteTimeGetCurrent() - startTime) * 1000)
            return ASLResult(
                predictedSign: "",
                confidence: 0.0,
                message: "Error: \(error.localizedDescription)",
                inferenceTimeMs: inferenceTime
            )
        }
    }
    
    /**
     * Recognize ASL sign from UIImage
     */
    func recognizeSign(from image: UIImage) -> ASLResult {
        guard let pixelBuffer = image.toPixelBuffer() else {
            return ASLResult(
                predictedSign: "",
                confidence: 0.0,
                message: "Failed to convert image",
                inferenceTimeMs: 0
            )
        }
        
        return recognizeSign(from: pixelBuffer)
    }
    
    /**
     * Get top N predictions
     */
    func getTopPredictions(from pixelBuffer: CVPixelBuffer, topN: Int = 5) -> [ASLResult] {
        guard isReady, let model = model else { return [] }
        
        do {
            guard let poseFeatures = poseExtractor.extractPoseFeatures(from: pixelBuffer) else {
                return []
            }
            
            let inputArray = try MLMultiArray(shape: [1, NSNumber(value: inputSize)], dataType: .float32)
            for (index, feature) in poseFeatures.enumerated() {
                if index < inputSize {
                    inputArray[index] = NSNumber(value: feature)
                }
            }
            
            let prediction = try model.prediction(input: inputArray)
            let probabilities = prediction.output
            
            // Create (index, confidence) pairs and sort by confidence
            var indexConfidencePairs: [(Int, Float)] = []
            for i in 0..<min(outputSize, probabilities.count) {
                indexConfidencePairs.append((i, probabilities[i].floatValue))
            }
            
            indexConfidencePairs.sort { $0.1 > $1.1 }  // Sort by confidence descending
            
            // Convert to ASLResults
            let topResults = indexConfidencePairs.prefix(topN).map { index, confidence in
                ASLResult(
                    predictedSign: getSignLabel(for: index),
                    confidence: confidence,
                    message: "Rank prediction",
                    inferenceTimeMs: 0
                )
            }
            
            return topResults
            
        } catch {
            print("❌ Top predictions error: \(error)")
            return []
        }
    }
    
    // MARK: - History Management
    
    /**
     * Add recognition to history
     */
    func addToHistory(_ result: ASLResult) {
        DispatchQueue.main.async { [weak self] in
            self?.recognitionHistory.append(result)
            
            // Keep only recent history (last 50 items)
            if let history = self?.recognitionHistory, history.count > 50 {
                self?.recognitionHistory = Array(history.suffix(50))
            }
        }
    }
    
    /**
     * Clear recognition history
     */
    func clearHistory() {
        DispatchQueue.main.async { [weak self] in
            self?.recognitionHistory.removeAll()
        }
    }
    
    /**
     * Build sentence from recent recognitions
     */
    func buildSentence() -> String {
        let recentResults = recognitionHistory.suffix(20)  // Last 20 recognitions
        var sentenceParts: [String] = []
        var currentSign: String?
        
        // Group consecutive similar signs
        for result in recentResults {
            if result.confidence > confidenceThreshold &&
               !result.predictedSign.isEmpty &&
               result.predictedSign != currentSign {
                
                sentenceParts.append(result.predictedSign)
                currentSign = result.predictedSign
            }
        }
        
        return sentenceParts.joined(separator: " ")
    }
    
    // MARK: - Utilities
    
    /**
     * Get sign label for index
     */
    private func getSignLabel(for index: Int) -> String {
        // Load labels from bundle or use default mapping
        if let labelsPath = Bundle.main.path(forResource: "asl_labels", ofType: "txt"),
           let labelsContent = try? String(contentsOfFile: labelsPath),
           !labelsContent.isEmpty {
            
            let labels = labelsContent.components(separatedBy: .newlines)
            return index < labels.count ? labels[index] : "UNKNOWN_\(index)"
        }
        
        // Fallback to generic labels
        return "SIGN_\(index)"
    }
    
    /**
     * Get model performance information
     */
    func getModelInfo() -> ModelInfo {
        return ModelInfo(
            modelPath: "ASLClassifier.mlmodel",
            inputSize: inputSize,
            outputSize: outputSize,
            vocabularySize: outputSize,
            quantization: "Core ML Optimized",
            status: isReady ? "Ready" : "Loading",
            device: getComputeDevice()
        )
    }
    
    private func getComputeDevice() -> String {
        guard let model = model else { return "Unknown" }
        
        // Try to determine compute device
        if #available(iOS 14.0, *) {
            // Check model configuration
            return "Neural Engine/GPU"
        } else {
            return "CPU"
        }
    }
}

// MARK: - Supporting Types

/**
 * ASL Recognition Result
 */
struct ASLResult: Identifiable, Codable {
    let id = UUID()
    let predictedSign: String
    let confidence: Float
    let message: String
    let inferenceTimeMs: Int64
    let timestamp: Date
    
    init(predictedSign: String, confidence: Float, message: String, inferenceTimeMs: Int64) {
        self.predictedSign = predictedSign
        self.confidence = confidence
        self.message = message
        self.inferenceTimeMs = inferenceTimeMs
        self.timestamp = Date()
    }
}

/**
 * Model Information
 */
struct ModelInfo {
    let modelPath: String
    let inputSize: Int
    let outputSize: Int
    let vocabularySize: Int
    let quantization: String
    let status: String
    let device: String
}

// MARK: - UIImage Extension

extension UIImage {
    /**
     * Convert UIImage to CVPixelBuffer
     */
    func toPixelBuffer() -> CVPixelBuffer? {
        let width = Int(self.size.width)
        let height = Int(self.size.height)
        
        let attrs = [
            kCVPixelBufferCGImageCompatibilityKey: kCFBooleanTrue,
            kCVPixelBufferCGBitmapContextCompatibilityKey: kCFBooleanTrue
        ] as CFDictionary
        
        var pixelBuffer: CVPixelBuffer?
        let status = CVPixelBufferCreate(
            kCFAllocatorDefault,
            width,
            height,
            kCVPixelFormatType_32ARGB,
            attrs,
            &pixelBuffer
        )
        
        guard status == kCVReturnSuccess, let buffer = pixelBuffer else {
            return nil
        }
        
        CVPixelBufferLockBaseAddress(buffer, CVPixelBufferLockFlags(rawValue: 0))
        let pixelData = CVPixelBufferGetBaseAddress(buffer)
        
        let colorSpace = CGColorSpaceCreateDeviceRGB()
        guard let context = CGContext(
            data: pixelData,
            width: width,
            height: height,
            bitsPerComponent: 8,
            bytesPerRow: CVPixelBufferGetBytesPerRow(buffer),
            space: colorSpace,
            bitmapInfo: CGImageAlphaInfo.noneSkipFirst.rawValue
        ) else {
            CVPixelBufferUnlockBaseAddress(buffer, CVPixelBufferLockFlags(rawValue: 0))
            return nil
        }
        
        context.translateBy(x: 0, y: CGFloat(height))
        context.scaleBy(x: 1, y: -1)
        
        UIGraphicsPushContext(context)
        self.draw(in: CGRect(x: 0, y: 0, width: width, height: height))
        UIGraphicsPopContext()
        
        CVPixelBufferUnlockBaseAddress(buffer, CVPixelBufferLockFlags(rawValue: 0))
        
        return buffer
    }
}
