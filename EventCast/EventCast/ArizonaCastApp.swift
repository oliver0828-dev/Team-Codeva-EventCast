//
//  ArizonaCastApp.swift
//  ArizonaCast
//
//  Created by Oliver Park on 10/5/25.
//

import SwiftUI
import SwiftData

@main
struct ArizonaCastApp: App {
    @StateObject private var settings = AppSettings()

    var body: some Scene {
        WindowGroup {
            AppRoot()                 // decides: onboarding or main tabs
                .environmentObject(settings)
        }
        .modelContainer(for: [UserEvent.self])
    }
}
