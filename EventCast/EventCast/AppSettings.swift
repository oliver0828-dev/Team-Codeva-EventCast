//
//  AppSettings.swift
//  ArizonaCast
//
//  Created by Oliver Park on 10/5/25.
//

import Foundation
import SwiftUI
import Combine

final class AppSettings: ObservableObject {
    let objectWillChange = ObservableObjectPublisher()
    @AppStorage("selectedCityRaw") private var selectedCityRaw: String = City.phoenix.rawValue
    @AppStorage("didCompleteOnboarding") var didCompleteOnboarding: Bool = false {
        didSet { objectWillChange.send() }
    }

    var selectedCity: City {
        get { City(rawValue: selectedCityRaw) ?? .phoenix }
        set { selectedCityRaw = newValue.rawValue; objectWillChange.send() }
    }
}

