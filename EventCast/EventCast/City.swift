//
//  City.swift
//  ArizonaCast
//
//  Created by Oliver Park on 10/5/25.
//

import Foundation
import Combine
import MapKit

enum City: String, CaseIterable, Identifiable {
    case phoenix = "Phoenix"
    case seoul = "Seoul"
    case tokyo = "Tokyo"
    case singapore = "Singapore"
    case delhi = "Delhi"

    var id: String { rawValue }

    var resourceName: String {
        switch self {
        case .phoenix:   return "weather_predictions_phoenix_with_feelings"
        case .seoul:     return "weather_predictions_seoul_with_feelings"
        case .tokyo:     return "weather_predictions_tokyo_with_feelings"
        case .singapore: return "weather_predictions_singapore_with_feelings"
        case .delhi:     return "weather_predictions_delhi_with_feelings"
        }
    }

    /// Simple “address contains” heuristic for gating
    func matches(address: String) -> Bool {
        let s = address.lowercased()
        switch self {
        case .phoenix:   return s.contains("phoenix") || s.contains("arizona") || s.hasSuffix("az")
        case .seoul:
            return s.contains("seoul") || address.contains("서울")
        case .tokyo:     return s.contains("tokyo") || s.contains("japan")
        case .singapore: return s.contains("singapore")
        case .delhi:     return s.contains("delhi") || s.contains("new delhi") || s.contains("india")
        }
    }

    var addressHint: String {
        switch self {
        case .phoenix:   return "Phoenix, Arizona"
        case .seoul:     return "Seoul"
        case .tokyo:     return "Tokyo, Japan"
        case .singapore: return "Singapore"
        case .delhi:     return "New Delhi, India"
        }
    }
    
    var coordinate: CLLocationCoordinate2D {
            switch self {
            case .phoenix:
                return CLLocationCoordinate2D(latitude: 33.4484, longitude: -112.0740)
            case .seoul:
                return CLLocationCoordinate2D(latitude: 37.5665, longitude: 126.9780)
            case .tokyo:
                return CLLocationCoordinate2D(latitude: 35.6762, longitude: 139.6503)
            case .singapore:
                return CLLocationCoordinate2D(latitude: 1.3521, longitude: 103.8198)
            case .delhi:
                return CLLocationCoordinate2D(latitude: 28.6139, longitude: 77.2090)
            }
        }

        /// Default map region (center + span) for this city
        var mapRegion: MKCoordinateRegion {
            let span: MKCoordinateSpan
            let center: CLLocationCoordinate2D
            switch self {
            case .phoenix:
                span = MKCoordinateSpan(latitudeDelta: 1.6, longitudeDelta: 1.6)
                center = coordinate
            case .seoul:
                span = MKCoordinateSpan(latitudeDelta: 1.0, longitudeDelta: 1.0)
                center = coordinate
            case .tokyo:
                span = MKCoordinateSpan(latitudeDelta: 0.9, longitudeDelta: 0.9)
                center = coordinate
            case .singapore:
                span = MKCoordinateSpan(latitudeDelta: 0.3, longitudeDelta: 0.3)
                center = coordinate
            case .delhi:
                span = MKCoordinateSpan(latitudeDelta: 1.2, longitudeDelta: 1.2)
                center = coordinate
            }
            return MKCoordinateRegion(center: center, span: span)
        }
}
