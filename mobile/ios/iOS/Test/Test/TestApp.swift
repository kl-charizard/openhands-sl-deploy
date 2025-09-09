//
//  TestApp.swift
//  Test
//
//  Created by Kenny Lam on 9/9/25.
//

import SwiftUI
import AVFoundation

@main
struct TestApp: App {
    var body: some Scene {
        WindowGroup {
            ContentView()
                .onAppear {
                    setupApp()
                }
        }
    }
    
    private func setupApp() {
        // Request camera permissions on app launch
        requestCameraPermission { granted in
            if granted {
                print("✅ Camera permission granted")
            } else {
                print("❌ Camera permission denied")
            }
        }
    }
    
    private func requestCameraPermission(completion: @escaping (Bool) -> Void) {
        AVCaptureDevice.requestAccess(for: .video) { granted in
            DispatchQueue.main.async {
                completion(granted)
            }
        }
    }
}

// MARK: - App Info and Configuration

extension TestApp {
    
    static var appVersion: String {
        Bundle.main.infoDictionary?["CFBundleShortVersionString"] as? String ?? "1.0"
    }
    
    static var buildNumber: String {
        Bundle.main.infoDictionary?["CFBundleVersion"] as? String ?? "1"
    }
    
    static var appName: String {
        Bundle.main.infoDictionary?["CFBundleDisplayName"] as? String ?? "ASL Recognition"
    }
}
