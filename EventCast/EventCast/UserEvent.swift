//
//  UserEvent.swift
//  ArizonaCast
//
//  Created by Oliver Park on 10/5/25.
//

import Foundation
import SwiftData

@Model
final class UserEvent {
    var title: String
    var date: Date
    var address: String
    var cityRaw: String        // persisted

    init(title: String, date: Date, address: String, city: City) {
        self.title = title
        self.date = date
        self.address = address
        self.cityRaw = city.rawValue
    }

    var city: City {
        City(rawValue: cityRaw).or(.phoenix)  // fallback
    }
}

private extension Optional where Wrapped == City {
    func or(_ fallback: City) -> City { self ?? fallback }
}
