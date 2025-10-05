//
//  RootTabs.swift
//  ArizonaCast
//
//  Created by Oliver Park on 10/5/25.
//

import SwiftUI
import Combine

struct RootTabs: View {
    @EnvironmentObject var settings: AppSettings
    @StateObject private var vmHolder = VMHolder()

    var body: some View {
        TabView {
            ContentView(vm: vmHolder.vm(settings.selectedCity))
                .tabItem { Label("Explore", systemImage: "chart.xyaxis.line") }

            EventsView(vm: vmHolder.vm(settings.selectedCity))
                .tabItem { Label("Events", systemImage: "calendar") }

            SettingsView()
                .tabItem { Label("Settings", systemImage: "gear") }
        }
        // if city changes in Settings, refresh the VM
        .onChange(of: settings.selectedCity) { _, newCity in
            vmHolder.replace(with: newCity)
        }
    }
}

/// Keeps a single VM instance, but allows replacement when city changes
final class VMHolder: ObservableObject {
    @Published private var _vm: WeatherViewModel?

    func vm(_ city: City) -> WeatherViewModel {
        if let v = _vm { return v }
        let v = WeatherViewModel(city: city)
        _vm = v
        return v
    }
    func replace(with city: City) {
        _vm = WeatherViewModel(city: city)
    }
}

