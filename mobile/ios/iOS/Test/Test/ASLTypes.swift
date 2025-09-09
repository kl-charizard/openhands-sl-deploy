import Foundation
import SwiftUI

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

/**
 * Motion Types
 */
enum MotionType: String, CaseIterable {
    case stationary = "Stationary"
    case waving = "Waving"  
    case pointing = "Pointing"
    case circular = "Circular"
    case upward = "Upward"
    case downward = "Downward"
    case sideways = "Sideways"
    case complex = "Complex"
    case rapid = "Rapid"
    
    var emoji: String {
        switch self {
        case .stationary: return "🚫"
        case .waving: return "👋"
        case .pointing: return "👉"
        case .circular: return "🔄"
        case .upward: return "⬆️"
        case .downward: return "⬇️"
        case .sideways: return "↔️"
        case .complex: return "🌪️"
        case .rapid: return "⚡"
        }
    }
    
    var description: String {
        switch self {
        case .stationary: return "Hand is still"
        case .waving: return "Waving motion detected"
        case .pointing: return "Pointing gesture"
        case .circular: return "Circular movement"
        case .upward: return "Moving upward"
        case .downward: return "Moving downward"
        case .sideways: return "Side to side movement"
        case .complex: return "Complex motion pattern"
        case .rapid: return "Rapid movement"
        }
    }
}

/**
 * Motion Summary
 */
struct MotionSummary {
    let currentMotion: MotionType
    let handCount: Int
    let bodyDetected: Bool
    let recentGestures: [String]
    let motionIntensity: Float
    let gestureCompleted: Bool
}
