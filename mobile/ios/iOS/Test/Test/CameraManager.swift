import Foundation
import AVFoundation
import Vision
import SwiftUI
import UIKit
import Combine

/**
 * 📱 Camera Manager with Body & Hand Motion Detection
 * Integrates with existing ASLRecognizer
 */
@available(iOS 14.0, *)
class CameraManager: NSObject, ObservableObject {
    
    // MARK: - Properties
    
    private var captureSession: AVCaptureSession?
    private var videoOutput: AVCaptureVideoDataOutput?
    private var previewLayer: AVCaptureVideoPreviewLayer?
    private let sessionQueue = DispatchQueue(label: "camera.session.queue")
    private let videoQueue = DispatchQueue(label: "camera.video.queue")
    
    // ASL Recognition
    private var aslRecognizer: ASLRecognizer?
    private var poseExtractor: PoseExtractor?
    
    // Published properties for UI
    @Published var isRecording = false
    @Published var currentResult: ASLResult?
    @Published var motionSummary: MotionSummary?
    @Published var motionType: MotionType = .stationary
    @Published var handsDetected = false
    @Published var bodyDetected = false
    
    // Performance tracking
    private var frameCount = 0
    private var lastFPSUpdate = Date()
    @Published var currentFPS: Double = 0.0
    
    // Landmark overlay
    private weak var landmarkOverlay: PoseLandmarkOverlayView?
    
    // MARK: - Setup
    
    func setup(with recognizer: ASLRecognizer) {
        self.aslRecognizer = recognizer
        self.poseExtractor = PoseExtractor()
        
        sessionQueue.async {
            self.setupCaptureSession()
        }
    }
    
    private func setupCaptureSession() {
        guard captureSession == nil else { return }
        
        // Check camera permissions first
        let authStatus = AVCaptureDevice.authorizationStatus(for: .video)
        guard authStatus == .authorized else {
            print("❌ Camera access not authorized: \(authStatus)")
            if authStatus == .notDetermined {
                AVCaptureDevice.requestAccess(for: .video) { granted in
                    if granted {
                        DispatchQueue.main.async {
                            self.setupCaptureSession()
                        }
                    }
                }
            }
            return
        }
        
        let session = AVCaptureSession()
        session.sessionPreset = .high
        
        // Add camera input
        guard let camera = AVCaptureDevice.default(.builtInWideAngleCamera, for: .video, position: .front),
              let input = try? AVCaptureDeviceInput(device: camera) else {
            print("❌ Failed to setup camera input")
            return
        }
        
        if session.canAddInput(input) {
            session.addInput(input)
        }
        
        // Add video output
        let output = AVCaptureVideoDataOutput()
        output.setSampleBufferDelegate(self, queue: videoQueue)
        output.videoSettings = [
            kCVPixelBufferPixelFormatTypeKey as String: kCVPixelFormatType_32BGRA
        ]
        
        if session.canAddOutput(output) {
            session.addOutput(output)
            self.videoOutput = output
        }
        
        self.captureSession = session
        
        // Start the session
        session.startRunning()
        
        print("✅ Camera setup completed and session started")
    }
    
    func setupPreview(in view: UIView) {
        sessionQueue.async {
            guard let session = self.captureSession else { 
                print("❌ No capture session available for preview")
                return 
            }
            
            DispatchQueue.main.async {
                // Clear any existing preview layers
                view.layer.sublayers?.forEach { layer in
                    if layer is AVCaptureVideoPreviewLayer {
                        layer.removeFromSuperlayer()
                    }
                }
                
                let previewLayer = AVCaptureVideoPreviewLayer(session: session)
                previewLayer.frame = view.bounds
                previewLayer.videoGravity = .resizeAspectFill
                previewLayer.backgroundColor = UIColor.black.cgColor
                
                view.layer.addSublayer(previewLayer)
                self.previewLayer = previewLayer
                
                print("✅ Preview layer added to view")
            }
        }
    }
    
    func setLandmarkOverlay(_ overlay: PoseLandmarkOverlayView) {
        self.landmarkOverlay = overlay
    }
    
    // MARK: - Recording Control
    
    func toggleRecording() {
        sessionQueue.async {
            if self.isRecording {
                self.stopRecording()
            } else {
                self.startRecording()
            }
        }
    }
    
    private func startRecording() {
        guard let session = captureSession, !session.isRunning else { return }
        
        session.startRunning()
        
        DispatchQueue.main.async {
            self.isRecording = true
        }
        
        print("🎥 Started camera recording")
    }
    
    private func stopRecording() {
        guard let session = captureSession, session.isRunning else { return }
        
        session.stopRunning()
        
        DispatchQueue.main.async {
            self.isRecording = false
        }
        
        print("⏹️ Stopped camera recording")
    }
    
    // MARK: - Detection Control
    
    func resetDetection() {
        poseExtractor?.resetTracking()
        
        DispatchQueue.main.async {
            self.currentResult = nil
            self.motionSummary = nil
            self.motionType = .stationary
            self.handsDetected = false
            self.bodyDetected = false
        }
    }
    
    // MARK: - Frame Processing
    
    private func processFrame(_ pixelBuffer: CVPixelBuffer) {
        // Update FPS counter
        updateFPS()
        
        // Extract pose features with motion detection
        guard let poseExtractor = self.poseExtractor else { return }
        
        // Get motion summary first
        let motionSummary = poseExtractor.getMotionSummary()
        
        // Update UI with motion info
        DispatchQueue.main.async {
            self.motionSummary = motionSummary
            self.motionType = motionSummary.currentMotion
            self.handsDetected = motionSummary.handCount > 0
            self.bodyDetected = motionSummary.bodyDetected
        }
        
        // Perform ASL recognition
        guard let aslRecognizer = self.aslRecognizer,
              aslRecognizer.isReady else { return }
        
        let result = aslRecognizer.recognizeSign(from: pixelBuffer)
        
        // Update UI with recognition result
        DispatchQueue.main.async {
            self.currentResult = result
        }
        
        // Auto-save significant results
        if result.confidence > 0.8 && !result.predictedSign.isEmpty {
            aslRecognizer.addToHistory(result)
        }
    }
    
    private func updateFPS() {
        frameCount += 1
        
        let now = Date()
        let timeDiff = now.timeIntervalSince(lastFPSUpdate)
        
        if timeDiff >= 1.0 { // Update every second
            let fps = Double(frameCount) / timeDiff
            
            DispatchQueue.main.async {
                self.currentFPS = fps
            }
            
            frameCount = 0
            lastFPSUpdate = now
        }
    }
}

// MARK: - AVCaptureVideoDataOutputSampleBufferDelegate

extension CameraManager: AVCaptureVideoDataOutputSampleBufferDelegate {
    
    func captureOutput(_ output: AVCaptureOutput, 
                      didOutput sampleBuffer: CMSampleBuffer, 
                      from connection: AVCaptureConnection) {
        
        // Convert sample buffer to pixel buffer
        guard let pixelBuffer = CMSampleBufferGetImageBuffer(sampleBuffer) else {
            return
        }
        
        // Process the frame
        processFrame(pixelBuffer)
    }
    
    func captureOutput(_ output: AVCaptureOutput, 
                      didDrop sampleBuffer: CMSampleBuffer, 
                      from connection: AVCaptureConnection) {
        // Handle dropped frames if needed
        print("⚠️ Dropped frame")
    }
}

// MARK: - Camera Permissions

extension CameraManager {
    
    static func checkCameraPermission() -> AVAuthorizationStatus {
        return AVCaptureDevice.authorizationStatus(for: .video)
    }
    
    static func requestCameraPermission(completion: @escaping (Bool) -> Void) {
        AVCaptureDevice.requestAccess(for: .video) { granted in
            DispatchQueue.main.async {
                completion(granted)
            }
        }
    }
}

// MARK: - Extensions for existing types

