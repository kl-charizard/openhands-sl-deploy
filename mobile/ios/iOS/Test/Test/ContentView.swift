//
//  ContentView.swift
//  Test
//
//  Created by Kenny Lam on 9/9/25.
//

import SwiftUI
import AVFoundation
import Vision

struct ContentView: View {
    @StateObject private var aslRecognizer = ASLRecognizer()
    @StateObject private var cameraManager = CameraManager()
    @State private var showingSettings = false
    @State private var isSetup = false
    
    var body: some View {
        NavigationView {
            ZStack {
                // Camera Preview
                if isSetup {
                    CameraPreviewView(cameraManager: cameraManager)
                        .ignoresSafeArea()
                } else {
                    Color.black
                        .ignoresSafeArea()
                        .overlay(
                            VStack {
                                ProgressView()
                                    .progressViewStyle(CircularProgressViewStyle(tint: .white))
                                    .scaleEffect(1.5)
                                Text("Initializing Camera...")
                                    .foregroundColor(.white)
                                    .padding(.top)
                            }
                        )
                }
                
                // Overlay UI
                VStack {
                    // Top Status Bar
                    HStack {
                        StatusIndicator(
                            title: "Model",
                            status: aslRecognizer.isReady ? "Ready" : "Loading",
                            color: aslRecognizer.isReady ? .green : .orange
                        )
                        
                        Spacer()
                        
                        StatusIndicator(
                            title: "Motion",
                            status: cameraManager.motionType.rawValue,
                            color: .blue
                        )
                        
                        Spacer()
                        
                        Button(action: { showingSettings = true }) {
                            Image(systemName: "gear")
                                .foregroundColor(.white)
                                .background(Color.black.opacity(0.5))
                                .clipShape(Circle())
                                .padding()
                        }
                    }
                    .padding(.top)
                    
                    Spacer()
                    
                    // Recognition Results
                    RecognitionResultsView(
                        currentResult: cameraManager.currentResult,
                        motionSummary: cameraManager.motionSummary
                    )
                    
                    Spacer()
                    
                    // Control Buttons
                    HStack(spacing: 20) {
                        Button("Clear History") {
                            aslRecognizer.clearHistory()
                            cameraManager.resetDetection()
                        }
                        .buttonStyle(ASLButtonStyle(color: .red))
                        
                        Button(cameraManager.isRecording ? "Stop" : "Start") {
                            cameraManager.toggleRecording()
                        }
                        .buttonStyle(ASLButtonStyle(color: cameraManager.isRecording ? .red : .green))
                        
                        Button("Save Result") {
                            if let result = cameraManager.currentResult {
                                aslRecognizer.addToHistory(result)
                            }
                        }
                        .buttonStyle(ASLButtonStyle(color: .blue))
                    }
                    .padding(.bottom, 50)
                }
            }
        }
        .navigationBarHidden(true)
        .onAppear {
            setupCameraManager()
        }
        .sheet(isPresented: $showingSettings) {
            SettingsView(aslRecognizer: aslRecognizer)
        }
    }
    
    private func setupCameraManager() {
        // Only setup once
        guard !isSetup else { return }
        
        print("🔄 Setting up camera manager...")
        
        // Setup camera manager with ASL recognizer
        cameraManager.setup(with: aslRecognizer)
        
        // Wait a moment for camera setup to complete, then show preview
        DispatchQueue.main.asyncAfter(deadline: .now() + 1.0) {
            self.isSetup = true
            print("✅ Camera manager setup completed, showing preview")
        }
    }
}

// MARK: - Camera Preview with Overlays
struct CameraPreviewView: UIViewRepresentable {
    let cameraManager: CameraManager
    
    func makeUIView(context: Context) -> UIView {
        let containerView = UIView(frame: UIScreen.main.bounds)
        containerView.backgroundColor = .black
        
        // Setup camera preview
        cameraManager.setupPreview(in: containerView)
        
        // Add overlay for hand and body landmarks
        let overlayView = PoseLandmarkOverlayView()
        overlayView.frame = containerView.bounds
        overlayView.backgroundColor = .clear
        containerView.addSubview(overlayView)
        
        // Connect overlay to camera manager for updates
        cameraManager.setLandmarkOverlay(overlayView)
        
        return containerView
    }
    
    func updateUIView(_ uiView: UIView, context: Context) {
        // Update preview layer frame when view size changes
        DispatchQueue.main.async {
            if let previewLayer = uiView.layer.sublayers?.first(where: { $0 is AVCaptureVideoPreviewLayer }) as? AVCaptureVideoPreviewLayer {
                previewLayer.frame = uiView.bounds
            }
            
            // Update overlay frame
            if let overlayView = uiView.subviews.first(where: { $0 is PoseLandmarkOverlayView }) {
                overlayView.frame = uiView.bounds
            }
        }
    }
}

// MARK: - Pose Landmark Overlay
class PoseLandmarkOverlayView: UIView {
    private var handLandmarks: [CGPoint] = []
    private var bodyLandmarks: [CGPoint] = []
    private var currentGesture: String = ""
    private var gestureConfidence: Float = 0.0
    
    override func draw(_ rect: CGRect) {
        super.draw(rect)
        
        guard let context = UIGraphicsGetCurrentContext() else { return }
        
        // Clear previous drawings
        context.clear(rect)
        
        // Draw hand landmarks
        drawHandLandmarks(context: context)
        
        // Draw body landmarks  
        drawBodyLandmarks(context: context)
        
        // Draw gesture information
        drawGestureInfo(context: context, rect: rect)
    }
    
    private func drawHandLandmarks(context: CGContext) {
        guard !handLandmarks.isEmpty else { return }
        
        // Set hand landmark style
        context.setStrokeColor(UIColor.cyan.cgColor)
        context.setFillColor(UIColor.cyan.withAlphaComponent(0.8).cgColor)
        context.setLineWidth(2.0)
        
        // Draw hand skeleton connections
        let handConnections = getHandConnections()
        for connection in handConnections {
            if connection.0 < handLandmarks.count && connection.1 < handLandmarks.count {
                let point1 = handLandmarks[connection.0]
                let point2 = handLandmarks[connection.1]
                
                context.move(to: point1)
                context.addLine(to: point2)
                context.strokePath()
            }
        }
        
        // Draw hand landmark points
        for landmark in handLandmarks {
            let rect = CGRect(x: landmark.x - 4, y: landmark.y - 4, width: 8, height: 8)
            context.fillEllipse(in: rect)
        }
    }
    
    private func drawBodyLandmarks(context: CGContext) {
        guard !bodyLandmarks.isEmpty else { return }
        
        // Set body landmark style
        context.setStrokeColor(UIColor.yellow.cgColor)
        context.setFillColor(UIColor.yellow.withAlphaComponent(0.8).cgColor)
        context.setLineWidth(3.0)
        
        // Draw body skeleton connections
        let bodyConnections = getBodyConnections()
        for connection in bodyConnections {
            if connection.0 < bodyLandmarks.count && connection.1 < bodyLandmarks.count {
                let point1 = bodyLandmarks[connection.0]
                let point2 = bodyLandmarks[connection.1]
                
                context.move(to: point1)
                context.addLine(to: point2)
                context.strokePath()
            }
        }
        
        // Draw body landmark points
        for landmark in bodyLandmarks {
            let rect = CGRect(x: landmark.x - 6, y: landmark.y - 6, width: 12, height: 12)
            context.fillEllipse(in: rect)
        }
    }
    
    private func drawGestureInfo(context: CGContext, rect: CGRect) {
        guard !currentGesture.isEmpty else { return }
        
        // Draw gesture text background
        let text = "\(currentGesture) (\(Int(gestureConfidence * 100))%)"
        let font = UIFont.boldSystemFont(ofSize: 24)
        let attributes: [NSAttributedString.Key: Any] = [
            .font: font,
            .foregroundColor: UIColor.white
        ]
        
        let textSize = text.size(withAttributes: attributes)
        let backgroundRect = CGRect(
            x: 20,
            y: rect.height - 80,
            width: textSize.width + 20,
            height: textSize.height + 10
        )
        
        context.setFillColor(UIColor.black.withAlphaComponent(0.7).cgColor)
        let roundedPath = UIBezierPath(roundedRect: backgroundRect, cornerRadius: 8)
        context.addPath(roundedPath.cgPath)
        context.fillPath()
        
        // Draw gesture text
        let textRect = CGRect(
            x: backgroundRect.minX + 10,
            y: backgroundRect.minY + 5,
            width: textSize.width,
            height: textSize.height
        )
        
        text.draw(in: textRect, withAttributes: attributes)
    }
    
    // Update landmarks from camera manager
    func updateHandLandmarks(_ landmarks: [CGPoint]) {
        handLandmarks = landmarks
        DispatchQueue.main.async {
            self.setNeedsDisplay()
        }
    }
    
    func updateBodyLandmarks(_ landmarks: [CGPoint]) {
        bodyLandmarks = landmarks
        DispatchQueue.main.async {
            self.setNeedsDisplay()
        }
    }
    
    func updateGesture(_ gesture: String, confidence: Float) {
        currentGesture = gesture
        gestureConfidence = confidence
        DispatchQueue.main.async {
            self.setNeedsDisplay()
        }
    }
    
    // Hand landmark connections (MediaPipe hand model)
    private func getHandConnections() -> [(Int, Int)] {
        return [
            // Thumb
            (0, 1), (1, 2), (2, 3), (3, 4),
            // Index finger
            (0, 5), (5, 6), (6, 7), (7, 8),
            // Middle finger
            (0, 9), (9, 10), (10, 11), (11, 12),
            // Ring finger
            (0, 13), (13, 14), (14, 15), (15, 16),
            // Pinky
            (0, 17), (17, 18), (18, 19), (19, 20),
            // Palm connections
            (5, 9), (9, 13), (13, 17)
        ]
    }
    
    // Body landmark connections (MediaPipe pose model)
    private func getBodyConnections() -> [(Int, Int)] {
        return [
            // Head and shoulders
            (0, 1), (1, 2), (2, 3), (3, 7),
            (0, 4), (4, 5), (5, 6), (6, 8),
            // Torso
            (9, 10), (11, 12),
            // Arms
            (11, 13), (13, 15), (15, 19), (15, 21),
            (12, 14), (14, 16), (16, 20), (16, 22),
            // Legs  
            (23, 24), (23, 25), (24, 26),
            (25, 27), (26, 28), (27, 29),
            (28, 30), (29, 31), (30, 32), (31, 32)
        ]
    }
}

// MARK: - Status Indicator
struct StatusIndicator: View {
    let title: String
    let status: String
    let color: Color
    
    var body: some View {
        VStack(spacing: 2) {
            Text(title)
                .font(.caption2)
                .foregroundColor(.white)
            
            Text(status)
                .font(.caption)
                .fontWeight(.semibold)
                .foregroundColor(color)
        }
        .padding(.horizontal, 8)
        .padding(.vertical, 4)
        .background(Color.black.opacity(0.5))
        .cornerRadius(8)
    }
}

// MARK: - Recognition Results
struct RecognitionResultsView: View {
    let currentResult: ASLResult?
    let motionSummary: MotionSummary?
    
    var body: some View {
        VStack(spacing: 16) {
            // Current Recognition
            if let result = currentResult {
                VStack(spacing: 8) {
                    Text(result.predictedSign.isEmpty ? "No Sign Detected" : result.predictedSign)
                        .font(.title)
                        .fontWeight(.bold)
                        .foregroundColor(result.predictedSign.isEmpty ? .gray : .white)
                    
                    Text("Confidence: \(String(format: "%.1f%%", result.confidence * 100))")
                        .font(.subheadline)
                        .foregroundColor(.white.opacity(0.8))
                    
                    Text("\(result.inferenceTimeMs)ms")
                        .font(.caption)
                        .foregroundColor(.white.opacity(0.6))
                }
                .padding()
                .background(Color.black.opacity(0.7))
                .cornerRadius(12)
            }
            
            // Motion Information
            if let motion = motionSummary {
                VStack(spacing: 6) {
                    HStack {
                        Text("👋 Motion:")
                        Text(motion.currentMotion.rawValue)
                            .fontWeight(.semibold)
                        
                        Spacer()
                        
                        Text("🖐️ Hands: \(motion.handCount)")
                    }
                    
                    if !motion.recentGestures.isEmpty {
                        HStack {
                            Text("🎭 Gestures:")
                            Text(motion.recentGestures.joined(separator: " → "))
                                .fontWeight(.medium)
                            Spacer()
                        }
                    }
                    
                    // Motion intensity bar
                    HStack {
                        Text("Intensity:")
                        
                        GeometryReader { geometry in
                            ZStack(alignment: .leading) {
                                Rectangle()
                                    .fill(Color.white.opacity(0.3))
                                    .frame(height: 4)
                                
                                Rectangle()
                                    .fill(Color.blue)
                                    .frame(width: geometry.size.width * CGFloat(motion.motionIntensity), 
                                           height: 4)
                            }
                            .cornerRadius(2)
                        }
                        .frame(height: 4)
                    }
                }
                .font(.caption)
                .foregroundColor(.white)
                .padding()
                .background(Color.black.opacity(0.6))
                .cornerRadius(10)
            }
        }
        .padding(.horizontal)
    }
}

// MARK: - Settings View
struct SettingsView: View {
    @ObservedObject var aslRecognizer: ASLRecognizer
    @Environment(\.presentationMode) var presentationMode
    
    var body: some View {
        NavigationView {
            List {
                Section("Model Information") {
                    let modelInfo = aslRecognizer.getModelInfo()
                    
                    SettingRow(title: "Status", value: modelInfo.status)
                    SettingRow(title: "Device", value: modelInfo.device)
                    SettingRow(title: "Input Size", value: "\(modelInfo.inputSize)")
                    SettingRow(title: "Vocabulary", value: "\(modelInfo.vocabularySize)")
                    SettingRow(title: "Quantization", value: modelInfo.quantization)
                }
                
                Section("Recognition History") {
                    Text("Recent Results: \(aslRecognizer.recognitionHistory.count)")
                    
                    if !aslRecognizer.recognitionHistory.isEmpty {
                        Button("View History") {
                            // Show history detail
                        }
                        
                        Button("Clear History") {
                            aslRecognizer.clearHistory()
                        }
                        .foregroundColor(.red)
                    }
                }
                
                Section("Built Sentence") {
                    let sentence = aslRecognizer.buildSentence()
                    Text(sentence.isEmpty ? "No sentence built yet" : sentence)
                        .italic()
                }
            }
            .navigationTitle("Settings")
            .toolbar {
                ToolbarItem(placement: .navigationBarTrailing) {
                    Button("Done") {
                        presentationMode.wrappedValue.dismiss()
                    }
                }
            }
        }
    }
    
}

struct SettingRow: View {
    let title: String
    let value: String
    
    var body: some View {
        HStack {
            Text(title)
            Spacer()
            Text(value)
                .foregroundColor(.secondary)
        }
    }
}

// MARK: - Button Style
struct ASLButtonStyle: ButtonStyle {
    let color: Color
    
    func makeBody(configuration: Configuration) -> some View {
        configuration.label
            .foregroundColor(.white)
            .padding(.horizontal, 16)
            .padding(.vertical, 8)
            .background(color.opacity(configuration.isPressed ? 0.7 : 1.0))
            .cornerRadius(8)
            .scaleEffect(configuration.isPressed ? 0.95 : 1.0)
            .animation(.easeInOut(duration: 0.1), value: configuration.isPressed)
    }
}

// MARK: - Preview
#Preview {
    ContentView()
}
