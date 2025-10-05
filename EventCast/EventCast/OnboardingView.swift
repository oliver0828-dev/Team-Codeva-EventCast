//
//  OnboardingView.swift
//  ArizonaCast
//
//  Created by Oliver Park on 10/5/25.
//

import SwiftUI

struct OnboardingView: View {
    @EnvironmentObject var settings: AppSettings
    @State private var choice: City = .phoenix

    var body: some View {
        NavigationStack {
            VStack(spacing: 24) {
                Text("Choose Your Location").font(.title).bold()
                Picker("City", selection: $choice) {
                    ForEach(City.allCases) { c in Text(c.rawValue).tag(c) }
                }
                .pickerStyle(.wheel)
                .frame(maxHeight: 180)

                Text("You can change this later in Settings.")
                    .font(.subheadline)
                    .foregroundStyle(.secondary)

                Button {
                    settings.selectedCity = choice
                    settings.didCompleteOnboarding = true
                } label: {
                    Text("Continue").bold().frame(maxWidth: .infinity).padding()
                }
                .buttonStyle(.borderedProminent)
                .padding(.horizontal)
            }
            .padding()
        }
        .onAppear { choice = settings.selectedCity }
    }
}
