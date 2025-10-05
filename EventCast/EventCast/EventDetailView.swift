//
//  EventDetailView.swift
//  ArizonaCast
//
//  Created by Oliver Park on 10/5/25.
//

import SwiftUI
import Charts
import MapKit
import CoreLocation

struct EventDetailView: View {
    let event: UserEvent
    @ObservedObject var vm: WeatherViewModel

    var body: some View {
        ScrollView {
            VStack(alignment: .leading, spacing: 12) {
                header
                Divider().padding(.vertical, 4)
                locationMap
                Divider().padding(.vertical, 4)
                rainSummary
                Divider().padding(.vertical, 4)
                monthlyTrend
            }
            .padding()
        }
        .navigationTitle(event.title)
    }

    private var header: some View {
        VStack(alignment: .leading, spacing: 4) {
            Text(event.title).font(.title2).bold()
            HStack(spacing: 8) {
                Image(systemName: "calendar")
                Text(event.date, style: .date)
            }.foregroundStyle(.secondary)
            HStack(spacing: 8) {
                Image(systemName: "mappin.and.ellipse")
                Text(event.address).lineLimit(1)
            }.foregroundStyle(.secondary)
        }
    }

    private var rainSummary: some View {
        let prob = vm.rainProbability(for: event.date, in: event.city) ?? 0
        let emoji = WeatherEmoji.rainIntensity(prob: prob)
        let feeling = WeatherEmoji.rainFeelingLevel(prob: prob)
        return VStack(alignment: .leading, spacing: 8) {
            Text("Rain Probability").font(.headline)
            HStack {
                Text(emoji).font(.largeTitle)
                Text("\(Int(round(prob)))%")
                    .font(.system(size: 40, weight: .heavy, design: .rounded))
                    .monospacedDigit()
            }
            Text(feeling).font(.subheadline)
            Text("Based on historical data for this date/month.")
                .font(.footnote).foregroundStyle(.secondary)
        }
    }

    private var locationMap: some View {
        VStack(alignment: .leading, spacing: 8) {
            Text("Location").font(.headline)
            AddressMapView(address: event.address)
                .frame(height: 220)
                .clipShape(RoundedRectangle(cornerRadius: 12))
        }
    }

    private var monthlyTrend: some View {
        let month = Calendar.current.component(.month, from: event.date)
        let points = vm.monthlyRainTrend(forMonth: month, in: event.city)
        return VStack(alignment: .leading, spacing: 8) {
            Text("Monthly Rain Trend (by Year)").font(.headline)
            if points.isEmpty {
                Text("No data available.").foregroundStyle(.secondary)
            } else {
                Chart(points, id: \.year) { p in
                    BarMark(x: .value("Year", p.year),
                            y: .value("Avg Probability %", p.avgProbability))
                }
                .frame(height: 220)
                .chartXAxis {
                    AxisMarks(values: .automatic) { v in
                        if let y = v.as(Int.self) { AxisGridLine(); AxisTick(); AxisValueLabel("\(y)") }
                    }
                }
                .chartYAxis { AxisMarks(position: .leading) }
            }
        }
    }
}

private struct AddressMapView: View {
    let address: String
    @State private var region: MKCoordinateRegion?
    @State private var coordinate: CLLocationCoordinate2D?
    @State private var errorMessage: String?

    var body: some View {
        ZStack(alignment: .topTrailing) {
            Group {
                if region != nil {
                    Map(coordinateRegion: Binding(get: { self.region! }, set: { self.region = $0 }), annotationItems: coordinate.map { [CoordinateItem(coordinate: $0)] } ?? []) { item in
                        MapMarker(coordinate: item.coordinate)
                    }
                } else {
                    ZStack {
                        Color(.secondarySystemBackground)
                        if let errorMessage {
                            Text(errorMessage).foregroundStyle(.secondary)
                        } else {
                            HStack(spacing: 8) {
                                ProgressView()
                                Text("Locating address…").foregroundStyle(.secondary)
                            }
                        }
                    }
                }
            }

            if let coordinate {
                Button {
                    openInMaps(coordinate: coordinate)
                } label: {
                    Label("Open in Maps", systemImage: "map")
                        .font(.footnote)
                        .padding(8)
                        .background(.ultraThinMaterial, in: Capsule())
                }
                .padding(8)
            }
        }
        .task(id: address) {
            await geocode()
        }
    }

    @MainActor
    private func updateRegion(to coord: CLLocationCoordinate2D) {
        self.coordinate = coord
        self.region = MKCoordinateRegion(center: coord, span: MKCoordinateSpan(latitudeDelta: 0.05, longitudeDelta: 0.05))
    }

    private func geocode() async {
        let geocoder = CLGeocoder()
        let locale = preferredLocale(for: address)
        let regionBias = biasRegion(for: address)
        let query = normalizeForSouthKorea(address)

        // 1) Attempt geocoding with regional bias and locale if available
        do {
            let placemarks: [CLPlacemark]
            if #available(iOS 16.0, *) {
                placemarks = try await geocoder.geocodeAddressString(query, in: regionBias, preferredLocale: locale)
            } else if let regionBias {
                placemarks = try await geocoder.geocodeAddressString(query, in: regionBias)
            } else {
                placemarks = try await geocoder.geocodeAddressString(query)
            }
            if let loc = placemarks.first?.location {
                await updateRegion(to: loc.coordinate)
                await MainActor.run { self.errorMessage = nil }
                return
            }
        } catch {
            // Continue to fallback search below
        }

        // 2) Fallback: Use MKLocalSearch to find a place matching the address
        do {
            let request = MKLocalSearch.Request()
            request.naturalLanguageQuery = query
            request.resultTypes = .address
            if let regionBias {
                let mkRegion = MKCoordinateRegion(
                    center: regionBias.center,
                    span: MKCoordinateSpan(latitudeDelta: 6.0, longitudeDelta: 6.0)
                )
                request.region = mkRegion
            }
            let search = MKLocalSearch(request: request)
            let response = try await search.start()
            if let item = response.mapItems.first {
                await updateRegion(to: item.placemark.coordinate)
                await MainActor.run { self.errorMessage = nil }
                return
            }
        } catch {
            // Continue to final attempt
        }

        // 3) Final attempt: explicitly use Korean country name to disambiguate South Korea
        if suggestsSouthKorea(address) {
            do {
                let altQueryKR = address.contains("대한민국") ? query : query + ", 대한민국"
                let placemarks: [CLPlacemark]
                if #available(iOS 16.0, *) {
                    placemarks = try await geocoder.geocodeAddressString(altQueryKR, in: regionBias, preferredLocale: Locale(identifier: "ko_KR"))
                } else if let regionBias {
                    placemarks = try await geocoder.geocodeAddressString(altQueryKR, in: regionBias)
                } else {
                    placemarks = try await geocoder.geocodeAddressString(altQueryKR)
                }
                if let loc = placemarks.first?.location {
                    await updateRegion(to: loc.coordinate)
                    await MainActor.run { self.errorMessage = nil }
                    return
                }
            } catch {
                // Fall through to error state
            }
        }

        await MainActor.run { self.errorMessage = "Could not locate this address." }

        // Helper: normalize ambiguous Korea references toward South Korea unless clearly North
        func normalizeForSouthKorea(_ input: String) -> String {
            let lower = input.lowercased()
            // If clearly North Korea, do not override
            if suggestsNorthKorea(input) { return input }
            // Already clearly South Korea
            if lower.contains("south korea") || lower.contains("republic of korea") || input.contains("대한민국") { return input }
            var result = input
            // Replace ambiguous "Korea" with "South Korea"
            if lower.contains("korea") && !lower.contains("north korea") {
                result = result.replacingOccurrences(of: "Korea", with: "South Korea", options: [.caseInsensitive])
            }
            // Replace ISO code KR when used as country suffix
            if lower.hasSuffix(", kr") {
                result = String(result.dropLast(4)) + ", South Korea"
            }
            // If contains Hangul but no explicit country, append the Korean country name
            let hasHangul = input.range(of: "\\p{Hangul}", options: .regularExpression) != nil
            if hasHangul && !lower.contains("south korea") && !lower.contains("republic of korea") && !input.contains("대한민국") {
                result += ", 대한민국"
            }
            return result
        }
    }

    private func suggestsSouthKorea(_ address: String) -> Bool {
        let lower = address.lowercased()
        if lower.contains("south korea") || lower.contains("republic of korea") { return true }
        // Generic "korea" implies South unless explicitly North
        if lower.contains("korea") && !lower.contains("north korea") { return true }
        // Hangul or common South Korean names
        if address.contains("대한민국") || address.contains("한국") || address.contains("서울") { return true }
        // Detect Hangul characters (assume South if not explicitly North)
        if address.range(of: "\\p{Hangul}", options: .regularExpression) != nil && !lower.contains("north korea") { return true }
        return false
    }

    private func suggestsNorthKorea(_ address: String) -> Bool {
        let lower = address.lowercased()
        if lower.contains("north korea") { return true }
        if address.contains("조선") || address.contains("북한") { return true }
        return false
    }

    private func preferredLocale(for address: String) -> Locale? {
        if suggestsSouthKorea(address) { return Locale(identifier: "ko_KR") }
        return nil
    }

    private func biasRegion(for address: String) -> CLCircularRegion? {
        guard suggestsSouthKorea(address) && !suggestsNorthKorea(address) else { return nil }
        // Bias around the geographic center of South Korea
        let center = CLLocationCoordinate2D(latitude: 36.5, longitude: 127.8)
        // ~700 km radius to cover the entire peninsula and Jeju
        return CLCircularRegion(center: center, radius: 700_000, identifier: "SouthKoreaBias")
    }

    private func openInMaps(coordinate: CLLocationCoordinate2D) {
        let placemark = MKPlacemark(coordinate: coordinate)
        let item = MKMapItem(placemark: placemark)
        item.name = address
        item.openInMaps()
    }
}

private struct CoordinateItem: Identifiable {
    let id = UUID()
    let coordinate: CLLocationCoordinate2D
}

