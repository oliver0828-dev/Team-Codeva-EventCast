//
//  WeatherRecord.swift
//  ArizonaCast
//
//  Created by Oliver Park on 10/5/25.
//

import Foundation

struct WeatherRecord: Codable, Identifiable {
    var id: String { "\(date)-\(conditionText)" }

    let city: String
    let date: String
    let month: Int
    let month_name: String
    let day: Int

    let condition: String
    let probability_percent: Double
    let feeling_level: String
    let feeling_description: String

    let historical_rate_percent: Double
    let confidence_lower_percent: Double
    let confidence_upper_percent: Double
    let years_of_data: Int

    var parsedDate: Date? { isoDateFormatter.date(from: date) }
    var conditionText: String { condition }
}

let isoDateFormatter: DateFormatter = {
    let f = DateFormatter()
    f.calendar = Calendar(identifier: .iso8601)
    f.locale = Locale(identifier: "en_US_POSIX")
    f.timeZone = TimeZone(secondsFromGMT: 0)
    f.dateFormat = "yyyy-MM-dd"
    return f
}()
