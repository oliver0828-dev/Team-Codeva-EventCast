//
//  SettingsView.swift
//  ArizonaCast
//
//  Created by Oliver Park on 10/5/25.
//

import SwiftUI

struct SettingsView: View {
    @EnvironmentObject var settings: AppSettings

    var body: some View {
        NavigationStack {
            Form {
                Section(header: Text("Location")) {
                    Picker("City", selection: Binding(
                        get: { settings.selectedCity },
                        set: { settings.selectedCity = $0 }
                    )) {
                        ForEach(City.allCases) { c in Text(c.rawValue).tag(c) }
                    }
                    Text("Tip: enter addresses like “\(settings.selectedCity.addressHint)”.")
                        .font(.footnote).foregroundStyle(.secondary)
                }
            }
            .navigationTitle("Settings")
        }
    }
}
