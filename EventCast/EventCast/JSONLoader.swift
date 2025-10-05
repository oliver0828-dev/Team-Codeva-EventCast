//
//  JSONLoader.swift
//  ArizonaCast
//
//  Created by Oliver Park on 10/5/25.
//

// JSONLoader.swift
import Foundation

enum JSONLoader {
    static func loadAllCities() throws -> [WeatherRecord] {
        var merged: [WeatherRecord] = []
        for c in City.allCases {
            if let url = Bundle.main.url(forResource: c.resourceName, withExtension: "json") {
                let data = try Data(contentsOf: url)
                let arr = try JSONDecoder().decode([WeatherRecord].self, from: data)
                merged.append(contentsOf: arr)
            }
        }
        // Keep only recognized city names (safety)
        let allowed = Set(City.allCases.map { $0.rawValue })
        return merged.filter { allowed.contains($0.city) }
    }
}
