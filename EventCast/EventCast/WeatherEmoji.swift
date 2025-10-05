//
//  WeatherEmoji.swift
//  ArizonaCast
//
//  Created by Oliver Park on 10/5/25.
//

import Foundation

enum WeatherEmoji {
    static func forCondition(_ condition: String) -> String {
        switch condition.lowercased() {
        case let s where s.contains("wet"): return "🌧️"
        case let s where s.contains("wind"): return "🌬️"
        case let s where s.contains("hot"): return "🔥"
        case let s where s.contains("cold"): return "🥶"
        case let s where s.contains("uncomfortable"): return "🥵"
        default: return "☁️"
        }
    }
    static func rainIntensity(prob: Double) -> String {
        switch prob {
        case ..<10:  return "☀️"
        case ..<30:  return "🌤️"
        case ..<60:  return "🌦️"
        case ..<85:  return "🌧️"
        default:     return "⛈️"
        }
    }
    static func rainFeelingLevel(prob: Double) -> String {
        switch prob {
        case ..<10:  return "Dry & comfy"
        case ..<30:  return "Mostly fine"
        case ..<60:  return "Chance of showers"
        case ..<85:  return "Likely wet"
        default:     return "Stormy"
        }
    }
}
