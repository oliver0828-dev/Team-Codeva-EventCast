//
//  AppRoot.swift
//  ArizonaCast
//
//  Created by Oliver Park on 10/5/25.
//

import SwiftUI

struct AppRoot: View {
    @EnvironmentObject var settings: AppSettings

    var body: some View {
        if settings.didCompleteOnboarding {
            RootTabs()
        } else {
            OnboardingView()
        }
    }
}
