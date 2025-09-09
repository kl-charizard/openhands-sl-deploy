import Foundation
import Vision
import CoreML
import UIKit
import AVFoundation
import Combine

/**
 * 🖐️ Advanced Pose & Hand Motion Extractor for ASL
 * Combines Vision framework with custom motion tracking
 */
@available(iOS 14.0, *)
class PoseExtractor: ObservableObject {
    
    // MARK: - Properties
    
    private var handPoseRequest: VNDetectHumanHandPoseRequest
    private var bodyPoseRequest: VNDetectHumanBodyPoseRequest
    private var motionTracker: MotionTracker
    private var gestureRecognizer: GestureSequenceRecognizer
    
    @Published var currentHandLandmarks: [VNHumanHandPoseObservation] = []
    @Published var currentBodyLandmarks: VNHumanBodyPoseObservation?
    @Published var detectedMotion: MotionType = .stationary
    @Published var gestureSequence: [String] = []
    
    private let maxDetectionHistory = 30  // frames
    private var handHistory: [[VNHumanHandPoseObservation]] = []
    private var bodyHistory: [VNHumanBodyPoseObservation?] = []
    
    // MARK: - Initialization
    
    init() {
        // Initialize hand pose detection
        self.handPoseRequest = VNDetectHumanHandPoseRequest()
        self.handPoseRequest.maximumHandCount = 2
        
        // Initialize body pose detection  
        self.bodyPoseRequest = VNDetectHumanBodyPoseRequest()
        
        // Initialize motion tracking
        self.motionTracker = MotionTracker()
        self.gestureRecognizer = GestureSequenceRecognizer()
        
        print("🖐️ PoseExtractor initialized with Vision framework")
    }
    
    // MARK: - Main Extraction Methods
    
    /**
     * Extract comprehensive pose features from pixel buffer
     */
    func extractPoseFeatures(from pixelBuffer: CVPixelBuffer,
                           onHandLandmarks: @escaping ([CGPoint]) -> Void = { _ in },
                           onBodyLandmarks: @escaping ([CGPoint]) -> Void = { _ in },
                           onGesture: @escaping (String, Float) -> Void = { _, _ in }) -> [Float]? {
        
        // Extract hand landmarks with visual feedback
        extractHandLandmarksWithCallback(from: pixelBuffer) { landmarks in
            onHandLandmarks(landmarks)
        }
        
        // Extract body landmarks with visual feedback
        extractBodyLandmarksWithCallback(from: pixelBuffer) { landmarks in
            onBodyLandmarks(landmarks)
        }
        
        // Recognize gesture with visual feedback
        recognizeGestureWithCallback(from: pixelBuffer) { gesture, confidence in
            onGesture(gesture, confidence)
        }
        
        // Original feature extraction for compatibility
        guard let handFeatures = extractHandFeatures(from: pixelBuffer),
              let bodyFeatures = extractBodyFeatures(from: pixelBuffer) else {
            return nil
        }
        
        // Combine features
        var allFeatures = handFeatures
        allFeatures.append(contentsOf: bodyFeatures)
        
        // Add motion features
        let motionFeatures = motionTracker.getMotionFeatures()
        allFeatures.append(contentsOf: motionFeatures)
        
        // Update history
        updateDetectionHistory()
        
        return allFeatures
    }
    
    // MARK: - Visual Feedback Methods
    
    private func extractHandLandmarksWithCallback(from frame: CVPixelBuffer, completion: @escaping ([CGPoint]) -> Void) {
        let request = VNDetectHumanHandPoseRequest { request, error in
            guard let observations = request.results as? [VNHumanHandPoseObservation] else {
                completion([])
                return
            }
            
            var handPoints: [CGPoint] = []
            
            for observation in observations {
                // Get all hand landmarks
                let handLandmarks: [VNHumanHandPoseObservation.JointName] = [
                    .wrist, .thumbCMC, .thumbMP, .thumbIP, .thumbTip,
                    .indexMCP, .indexPIP, .indexDIP, .indexTip,
                    .middleMCP, .middlePIP, .middleDIP, .middleTip,
                    .ringMCP, .ringPIP, .ringDIP, .ringTip,
                    .littleMCP, .littlePIP, .littleDIP, .littleTip
                ]
                
                for jointName in handLandmarks {
                    if let point = try? observation.recognizedPoint(jointName) {
                        // Convert from normalized coordinates to screen coordinates
                        let screenPoint = CGPoint(
                            x: point.location.x * UIScreen.main.bounds.width,
                            y: (1.0 - point.location.y) * UIScreen.main.bounds.height
                        )
                        handPoints.append(screenPoint)
                    }
                }
                
                break // Only process first hand for now
            }
            
            DispatchQueue.main.async {
                completion(handPoints)
            }
        }
        
        let handler = VNImageRequestHandler(cvPixelBuffer: frame, options: [:])
        try? handler.perform([request])
    }
    
    private func extractBodyLandmarksWithCallback(from frame: CVPixelBuffer, completion: @escaping ([CGPoint]) -> Void) {
        let request = VNDetectHumanBodyPoseRequest { request, error in
            guard let observations = request.results as? [VNHumanBodyPoseObservation] else {
                completion([])
                return
            }
            
            var bodyPoints: [CGPoint] = []
            
            for observation in observations {
                // Get key body landmarks
                let bodyLandmarks: [VNHumanBodyPoseObservation.JointName] = [
                    .nose, .leftEye, .rightEye, .leftEar, .rightEar,
                    .leftShoulder, .rightShoulder, .leftElbow, .rightElbow,
                    .leftWrist, .rightWrist, .leftHip, .rightHip,
                    .leftKnee, .rightKnee, .leftAnkle, .rightAnkle
                ]
                
                for jointName in bodyLandmarks {
                    if let point = try? observation.recognizedPoint(jointName) {
                        // Convert from normalized coordinates to screen coordinates
                        let screenPoint = CGPoint(
                            x: point.location.x * UIScreen.main.bounds.width,
                            y: (1.0 - point.location.y) * UIScreen.main.bounds.height
                        )
                        bodyPoints.append(screenPoint)
                    }
                }
                
                break // Only process first person for now
            }
            
            DispatchQueue.main.async {
                completion(bodyPoints)
            }
        }
        
        let handler = VNImageRequestHandler(cvPixelBuffer: frame, options: [:])
        try? handler.perform([request])
    }
    
    private func recognizeGestureWithCallback(from frame: CVPixelBuffer, completion: @escaping (String, Float) -> Void) {
        // Simple gesture recognition based on hand pose
        let request = VNDetectHumanHandPoseRequest { request, error in
            guard let observations = request.results as? [VNHumanHandPoseObservation],
                  let observation = observations.first else {
                completion("", 0.0)
                return
            }
            
            // Simple gesture recognition logic
            let gesture = self.classifyHandGesture(observation)
            let confidence = Float.random(in: 0.7...0.95) // Mock confidence for now
            
            DispatchQueue.main.async {
                completion(gesture, confidence)
            }
        }
        
        let handler = VNImageRequestHandler(cvPixelBuffer: frame, options: [:])
        try? handler.perform([request])
    }
    
    private func classifyHandGesture(_ observation: VNHumanHandPoseObservation) -> String {
        // Simple heuristic-based gesture classification
        do {
            let indexTip = try observation.recognizedPoint(.indexTip)
            let indexMCP = try observation.recognizedPoint(.indexMCP)
            let thumbTip = try observation.recognizedPoint(.thumbTip)
            let thumbCMC = try observation.recognizedPoint(.thumbCMC)
            
            // Check if fingers are extended
            let indexExtended = indexTip.location.y < indexMCP.location.y
            let thumbExtended = thumbTip.location.x > thumbCMC.location.x
            
            if indexExtended && !thumbExtended {
                return "POINTING ☝️"
            } else if indexExtended && thumbExtended {
                return "PEACE ✌️"
            } else if !indexExtended && thumbExtended {
                return "THUMBS_UP 👍"
            } else {
                return "CLOSED_FIST ✊"
            }
        } catch {
            return "UNKNOWN"
        }
    }
    
    /**
     * Extract hand landmarks and features
     */
    func extractHandFeatures(from pixelBuffer: CVPixelBuffer) -> [Float]? {
        let requestHandler = VNImageRequestHandler(cvPixelBuffer: pixelBuffer, options: [:])
        
        do {
            try requestHandler.perform([handPoseRequest])
            
            guard let observations = handPoseRequest.results,
                  !observations.isEmpty else {
                return generateEmptyHandFeatures()
            }
            
            DispatchQueue.main.async {
                self.currentHandLandmarks = observations
            }
            
            return processHandObservations(observations)
            
        } catch {
            print("❌ Hand pose detection failed: \(error)")
            return generateEmptyHandFeatures()
        }
    }
    
    /**
     * Extract body pose landmarks and features
     */
    func extractBodyFeatures(from pixelBuffer: CVPixelBuffer) -> [Float]? {
        let requestHandler = VNImageRequestHandler(cvPixelBuffer: pixelBuffer, options: [:])
        
        do {
            try requestHandler.perform([bodyPoseRequest])
            
            guard let observations = bodyPoseRequest.results,
                  let bodyObservation = observations.first else {
                return generateEmptyBodyFeatures()
            }
            
            DispatchQueue.main.async {
                self.currentBodyLandmarks = bodyObservation
            }
            
            return processBodyObservation(bodyObservation)
            
        } catch {
            print("❌ Body pose detection failed: \(error)")
            return generateEmptyBodyFeatures()
        }
    }
    
    // MARK: - Hand Processing
    
    private func processHandObservations(_ observations: [VNHumanHandPoseObservation]) -> [Float] {
        var features: [Float] = []
        
        // Process up to 2 hands
        for (handIndex, observation) in observations.prefix(2).enumerated() {
            let handFeatures = extractSingleHandFeatures(observation, handIndex: handIndex)
            features.append(contentsOf: handFeatures)
        }
        
        // Pad if less than 2 hands detected
        while features.count < 84 { // 42 features per hand × 2 hands
            features.append(0.0)
        }
        
        // Update motion tracking
        motionTracker.updateHandMotion(observations)
        
        return features
    }
    
    private func extractSingleHandFeatures(_ observation: VNHumanHandPoseObservation, handIndex: Int) -> [Float] {
        var features: [Float] = []
        
        do {
            // Get all hand landmark points
            let landmarkKeys: [VNHumanHandPoseObservation.JointName] = [
                .wrist,
                .thumbCMC, .thumbMP, .thumbIP, .thumbTip,
                .indexMCP, .indexPIP, .indexDIP, .indexTip,
                .middleMCP, .middlePIP, .middleDIP, .middleTip,
                .ringMCP, .ringPIP, .ringDIP, .ringTip,
                .littleMCP, .littlePIP, .littleDIP, .littleTip
            ]
            
            // Extract coordinates for each landmark
            for jointName in landmarkKeys {
                if let point = try? observation.recognizedPoint(jointName) {
                    features.append(Float(point.location.x))
                    features.append(Float(point.location.y))
                } else {
                    features.append(0.0)
                    features.append(0.0)
                }
            }
            
            // Add hand shape analysis
            let handShape = analyzeHandShape(observation)
            features.append(contentsOf: handShape)
            
        } catch {
            print("❌ Failed to extract hand landmarks: \(error)")
            // Return zero-filled array
            return Array(repeating: 0.0, count: 42)
        }
        
        return features
    }
    
    private func analyzeHandShape(_ observation: VNHumanHandPoseObservation) -> [Float] {
        var shapeFeatures: [Float] = []
        
        do {
            // Finger extension analysis
            let fingerExtensions = analyzeFingersExtension(observation)
            shapeFeatures.append(contentsOf: fingerExtensions)
            
            // Hand orientation
            if let wrist = try? observation.recognizedPoint(.wrist),
               let middleMCP = try? observation.recognizedPoint(.middleMCP) {
                let angle = calculateAngle(from: wrist.location, to: middleMCP.location)
                shapeFeatures.append(Float(angle))
            } else {
                shapeFeatures.append(0.0)
            }
            
            // Hand size estimation
            let handSize = estimateHandSize(observation)
            shapeFeatures.append(handSize)
            
        } catch {
            shapeFeatures = Array(repeating: 0.0, count: 7) // 5 fingers + angle + size
        }
        
        return shapeFeatures
    }
    
    private func analyzeFingersExtension(_ observation: VNHumanHandPoseObservation) -> [Float] {
        var extensions: [Float] = []
        
        let fingers: [(tip: VNHumanHandPoseObservation.JointName, 
                      pip: VNHumanHandPoseObservation.JointName)] = [
            (.thumbTip, .thumbIP),
            (.indexTip, .indexPIP),
            (.middleTip, .middlePIP),
            (.ringTip, .ringPIP),
            (.littleTip, .littlePIP)
        ]
        
        for finger in fingers {
            do {
                if let tip = try? observation.recognizedPoint(finger.tip),
                   let pip = try? observation.recognizedPoint(finger.pip) {
                    // Simple extension check: tip above PIP
                    let isExtended = tip.location.y < pip.location.y
                    extensions.append(isExtended ? 1.0 : 0.0)
                } else {
                    extensions.append(0.0)
                }
            }
        }
        
        return extensions
    }
    
    private func estimateHandSize(_ observation: VNHumanHandPoseObservation) -> Float {
        do {
            if let wrist = try? observation.recognizedPoint(.wrist),
               let middleTip = try? observation.recognizedPoint(.middleTip) {
                let distance = sqrt(pow(middleTip.location.x - wrist.location.x, 2) +
                                  pow(middleTip.location.y - wrist.location.y, 2))
                return Float(distance)
            }
        } catch {}
        
        return 0.0
    }
    
    // MARK: - Body Processing
    
    private func processBodyObservation(_ observation: VNHumanBodyPoseObservation) -> [Float] {
        var features: [Float] = []
        
        do {
            // Key body points for ASL context
            let bodyKeys: [VNHumanBodyPoseObservation.JointName] = [
                .nose, .neck,
                .leftShoulder, .rightShoulder,
                .leftElbow, .rightElbow,
                .leftWrist, .rightWrist,
                .root
            ]
            
            // Extract body landmark coordinates
            for jointName in bodyKeys {
                if let point = try? observation.recognizedPoint(jointName) {
                    features.append(Float(point.location.x))
                    features.append(Float(point.location.y))
                } else {
                    features.append(0.0)
                    features.append(0.0)
                }
            }
            
            // Add body posture analysis
            let postureFeatures = analyzeBodyPosture(observation)
            features.append(contentsOf: postureFeatures)
            
        } catch {
            print("❌ Failed to process body observation: \(error)")
            features = Array(repeating: 0.0, count: 25) // 9 joints × 2 + posture features
        }
        
        // Update motion tracking for body
        motionTracker.updateBodyMotion(observation)
        
        return features
    }
    
    private func analyzeBodyPosture(_ observation: VNHumanBodyPoseObservation) -> [Float] {
        var postureFeatures: [Float] = []
        
        do {
            // Shoulder alignment
            if let leftShoulder = try? observation.recognizedPoint(.leftShoulder),
               let rightShoulder = try? observation.recognizedPoint(.rightShoulder) {
                let shoulderAngle = calculateAngle(from: leftShoulder.location, 
                                                 to: rightShoulder.location)
                postureFeatures.append(Float(shoulderAngle))
            } else {
                postureFeatures.append(0.0)
            }
            
            // Arm positions relative to body
            if let neck = try? observation.recognizedPoint(.neck),
               let leftWrist = try? observation.recognizedPoint(.leftWrist),
               let rightWrist = try? observation.recognizedPoint(.rightWrist) {
                
                // Left arm elevation
                let leftElevation = leftWrist.location.y - neck.location.y
                postureFeatures.append(Float(leftElevation))
                
                // Right arm elevation  
                let rightElevation = rightWrist.location.y - neck.location.y
                postureFeatures.append(Float(rightElevation))
                
                // Arms crossing detection
                let armsCrossed = (leftWrist.location.x > rightWrist.location.x) ? 1.0 : 0.0
                postureFeatures.append(Float(armsCrossed))
            } else {
                postureFeatures.append(contentsOf: [0.0, 0.0, 0.0])
            }
            
            // Upper body lean
            if let neck = try? observation.recognizedPoint(.neck),
               let root = try? observation.recognizedPoint(.root) {
                let bodyLean = neck.location.x - root.location.x
                postureFeatures.append(Float(bodyLean))
            } else {
                postureFeatures.append(0.0)
            }
            
        } catch {
            postureFeatures = Array(repeating: 0.0, count: 5)
        }
        
        return postureFeatures
    }
    
    // MARK: - Motion Analysis
    
    private func updateDetectionHistory() {
        // Store current detections in history
        handHistory.append(currentHandLandmarks)
        bodyHistory.append(currentBodyLandmarks)
        
        // Maintain history size
        if handHistory.count > maxDetectionHistory {
            handHistory.removeFirst()
        }
        if bodyHistory.count > maxDetectionHistory {
            bodyHistory.removeFirst()
        }
        
        // Update motion detection
        DispatchQueue.main.async {
            self.detectedMotion = self.motionTracker.getCurrentMotion()
        }
        
        // Update gesture sequence recognition
        let currentGesture = gestureRecognizer.analyzeSequence(
            handHistory: handHistory,
            bodyHistory: bodyHistory,
            motion: detectedMotion
        )
        
        if !currentGesture.isEmpty {
            DispatchQueue.main.async {
                self.gestureSequence.append(currentGesture)
                
                // Keep recent gesture sequence
                if self.gestureSequence.count > 10 {
                    self.gestureSequence.removeFirst()
                }
            }
        }
    }
    
    // MARK: - Utility Methods
    
    private func generateEmptyHandFeatures() -> [Float] {
        return Array(repeating: 0.0, count: 84) // 2 hands × 42 features each
    }
    
    private func generateEmptyBodyFeatures() -> [Float] {
        return Array(repeating: 0.0, count: 25) // 9 body joints × 2 + posture features
    }
    
    private func calculateAngle(from point1: CGPoint, to point2: CGPoint) -> Double {
        let deltaX = point2.x - point1.x
        let deltaY = point2.y - point1.y
        return atan2(deltaY, deltaX)
    }
    
    /**
     * Get comprehensive motion summary
     */
    func getMotionSummary() -> MotionSummary {
        return MotionSummary(
            currentMotion: detectedMotion,
            handCount: currentHandLandmarks.count,
            bodyDetected: currentBodyLandmarks != nil,
            recentGestures: Array(gestureSequence.suffix(5)),
            motionIntensity: motionTracker.getMotionIntensity(),
            gestureCompleted: gestureRecognizer.isGestureComplete()
        )
    }
    
    /**
     * Reset all tracking history
     */
    func resetTracking() {
        handHistory.removeAll()
        bodyHistory.removeAll()
        motionTracker.reset()
        gestureRecognizer.reset()
        
        DispatchQueue.main.async {
            self.currentHandLandmarks.removeAll()
            self.currentBodyLandmarks = nil
            self.detectedMotion = .stationary
            self.gestureSequence.removeAll()
        }
    }
}

// MARK: - Supporting Types


/**
 * Motion Tracker - Analyzes movement patterns
 */
class MotionTracker {
    private var handPositionHistory: [[CGPoint]] = []
    private var bodyPositionHistory: [CGPoint] = []
    private let maxHistorySize = 20
    private var currentMotion: MotionType = .stationary
    
    func updateHandMotion(_ observations: [VNHumanHandPoseObservation]) {
        var currentPositions: [CGPoint] = []
        
        for observation in observations {
            if let wrist = try? observation.recognizedPoint(.wrist) {
                currentPositions.append(wrist.location)
            }
        }
        
        handPositionHistory.append(currentPositions)
        
        if handPositionHistory.count > maxHistorySize {
            handPositionHistory.removeFirst()
        }
        
        analyzeHandMotion()
    }
    
    func updateBodyMotion(_ observation: VNHumanBodyPoseObservation) {
        if let root = try? observation.recognizedPoint(.root) {
            bodyPositionHistory.append(root.location)
            
            if bodyPositionHistory.count > maxHistorySize {
                bodyPositionHistory.removeFirst()
            }
        }
    }
    
    private func analyzeHandMotion() {
        guard handPositionHistory.count >= 5 else {
            currentMotion = .stationary
            return
        }
        
        // Analyze recent movement patterns
        let recentPositions = handPositionHistory.suffix(5)
        var totalMovement: CGFloat = 0
        var movementDirection: CGPoint = .zero
        
        for i in 1..<recentPositions.count {
            let prev = recentPositions[i-1]
            let curr = recentPositions[i]
            
            for j in 0..<min(prev.count, curr.count) {
                let movement = CGPoint(x: curr[j].x - prev[j].x, 
                                     y: curr[j].y - prev[j].y)
                totalMovement += sqrt(movement.x * movement.x + movement.y * movement.y)
                movementDirection.x += movement.x
                movementDirection.y += movement.y
            }
        }
        
        // Classify motion
        if totalMovement < 0.01 {
            currentMotion = .stationary
        } else if totalMovement > 0.1 {
            currentMotion = .rapid
        } else if abs(movementDirection.y) > abs(movementDirection.x) {
            currentMotion = movementDirection.y > 0 ? .downward : .upward
        } else {
            currentMotion = .sideways
        }
    }
    
    func getCurrentMotion() -> MotionType {
        return currentMotion
    }
    
    func getMotionFeatures() -> [Float] {
        return [
            Float(currentMotion.rawValue.count), // Simple encoding
            getMotionIntensity(),
            Float(handPositionHistory.count)
        ]
    }
    
    func getMotionIntensity() -> Float {
        guard handPositionHistory.count >= 2 else { return 0.0 }
        
        let recent = handPositionHistory.suffix(2)
        if recent.count >= 2 {
            let prev = recent.first!
            let curr = recent.last!
            
            var intensity: Float = 0.0
            for i in 0..<min(prev.count, curr.count) {
                let movement = sqrt(pow(curr[i].x - prev[i].x, 2) + 
                                  pow(curr[i].y - prev[i].y, 2))
                intensity += Float(movement)
            }
            
            return intensity
        }
        
        return 0.0
    }
    
    func reset() {
        handPositionHistory.removeAll()
        bodyPositionHistory.removeAll()
        currentMotion = .stationary
    }
}

/**
 * Gesture Sequence Recognizer - Recognizes gesture patterns over time
 */
class GestureSequenceRecognizer {
    private var gestureBuffer: [String] = []
    private let maxBufferSize = 15
    private var lastGestureTime = Date()
    private let gestureTimeout: TimeInterval = 3.0
    
    func analyzeSequence(handHistory: [[VNHumanHandPoseObservation]], 
                        bodyHistory: [VNHumanBodyPoseObservation?],
                        motion: MotionType) -> String {
        
        guard !handHistory.isEmpty else { return "" }
        
        // Simple gesture pattern recognition
        let currentTime = Date()
        if currentTime.timeIntervalSince(lastGestureTime) > gestureTimeout {
            gestureBuffer.removeAll()
        }
        
        // Analyze current frame for gesture components
        let gestureComponent = analyzeCurrentFrame(handHistory.last ?? [], 
                                                  bodyHistory.last ?? nil, 
                                                  motion)
        
        if !gestureComponent.isEmpty {
            gestureBuffer.append(gestureComponent)
            lastGestureTime = currentTime
            
            if gestureBuffer.count > maxBufferSize {
                gestureBuffer.removeFirst()
            }
            
            // Check for complete gesture patterns
            return recognizeGesturePattern()
        }
        
        return ""
    }
    
    private func analyzeCurrentFrame(_ hands: [VNHumanHandPoseObservation],
                                   _ body: VNHumanBodyPoseObservation?,
                                   _ motion: MotionType) -> String {
        
        guard !hands.isEmpty else { return "" }
        
        // Simple frame analysis
        switch motion {
        case .waving:
            return "WAVE"
        case .pointing:
            return "POINT"
        case .circular:
            return "CIRCLE"
        case .upward:
            return "UP"
        case .downward:
            return "DOWN"
        default:
            return ""
        }
    }
    
    private func recognizeGesturePattern() -> String {
        let pattern = gestureBuffer.joined(separator: "-")
        
        // Simple pattern matching
        if pattern.contains("UP-DOWN") {
            return "NOD"
        } else if pattern.contains("WAVE-WAVE") {
            return "GOODBYE"
        } else if pattern.contains("POINT") {
            return "INDICATE"
        } else if pattern.contains("CIRCLE") {
            return "PLEASE"
        }
        
        return ""
    }
    
    func isGestureComplete() -> Bool {
        return !gestureBuffer.isEmpty && gestureBuffer.count >= 2
    }
    
    func reset() {
        gestureBuffer.removeAll()
        lastGestureTime = Date()
    }
}
