//
//  WeatherViewModel.swift
//  ArizonaCast
//
//  Created by Oliver Park on 10/5/25.
//

import Foundation
import Combine

enum SelectionMode: String, CaseIterable, Identifiable {
    case month = "Month"
    case day   = "Specific Date"
    var id: String { rawValue }
}

final class WeatherViewModel: ObservableObject {
    // Inputs (Explore tab)
    @Published var address: String = ""
    @Published var mode: SelectionMode = .month
    @Published var selectedMonth: Int
    @Published var selectedDate: Date
    @Published var city: City { didSet { /* Explore swaps active city; data is already loaded for ALL */ } }

    // Data: merged across all cities
    @Published private(set) var all: [WeatherRecord] = []

    // Dataset bounds
    private let hardMin = isoDateFormatter.date(from: "2019-01-01")!
    private let hardMax = isoDateFormatter.date(from: "2024-12-31")!

    // Calendar pinned to GMT/ISO so month extraction is stable
    private let gmtCal: Calendar = {
        var c = Calendar(identifier: .iso8601)
        c.timeZone = TimeZone(secondsFromGMT: 0)!
        return c
    }()
    private func month(of d: Date) -> Int { gmtCal.component(.month, from: d) }
    private func year(of d: Date)  -> Int { gmtCal.component(.year,  from: d) }

    init(city: City) {
        self.city = city
        let now = Date()
        self.selectedMonth = Calendar.current.component(.month, from: now)
        self.selectedDate  = now
        load()
    }

    // MARK: Load ALL cities once
    func load() {
        do { all = try JSONLoader.loadAllCities() }
        catch { print("JSON load error:", error) }
    }

    // MARK: City filters
    private func data(for city: City) -> [WeatherRecord] {
        all.filter { $0.city == city.rawValue }
    }

    // Address gate for Explore (UI-only)
    var addressMatchesCity: Bool { city.matches(address: address) }

    // -------- Explore: Month mode (uses active city + gate) --------
    var monthModeInRangeRecords: [WeatherRecord] {
        guard addressMatchesCity else { return [] }
        return data(for: city)
            .filter { rec in
                guard let d = rec.parsedDate else { return false }
                return month(of: d) == selectedMonth && d >= hardMin && d <= hardMax
            }
            .sorted { ($0.parsedDate ?? .distantPast) < ($1.parsedDate ?? .distantPast) }
    }

    var monthModeAutoCondition: String? {
        topCondition(byAverageProbability: monthModeInRangeRecords)
    }

    var monthModeChartSeries: [WeatherRecord] {
        guard let cond = monthModeAutoCondition else { return [] }
        return monthModeInRangeRecords.filter { $0.conditionText == cond }
    }

    // -------- Explore: Day mode (uses active city + gate) --------
    private var dayYearInDataset: Bool {
        let yr = gmtCal.component(.year, from: selectedDate)
        let minYr = gmtCal.component(.year, from: hardMin)
        let maxYr = gmtCal.component(.year, from: hardMax)
        return (minYr...maxYr).contains(yr)
    }

    var dayExactRecordsInRange: [WeatherRecord] {
        guard addressMatchesCity, dayYearInDataset else { return [] }
        let target = isoDateFormatter.string(from: selectedDate)
        return data(for: city)
            .filter { $0.date == target }
            .sorted { $0.conditionText < $1.conditionText }
    }

    var dayModeAutoConditionInRange: String? {
        dayExactRecordsInRange.max(by: { $0.probability_percent < $1.probability_percent })?.conditionText
    }

    // -------- Explore: Month progression fallback --------
    private var dayModeSelectedMonth: Int { month(of: selectedDate) }

    private var monthProgressionRecords: [WeatherRecord] {
        guard addressMatchesCity else { return [] }
        return data(for: city)
            .filter { rec in
                guard let d = rec.parsedDate else { return false }
                return month(of: d) == dayModeSelectedMonth && d >= hardMin && d <= hardMax
            }
            .sorted { ($0.parsedDate ?? .distantPast) < ($1.parsedDate ?? .distantPast) }
    }

    var autoConditionForMonthProgression: String? {
        topCondition(byAverageProbability: monthProgressionRecords)
    }

    var monthProgressionPerYear: [(year: Int, avgProbability: Double, samples: [WeatherRecord])] {
        guard let cond = autoConditionForMonthProgression else { return [] }
        let filtered = monthProgressionRecords.filter { $0.conditionText == cond }
        let grouped = Dictionary(grouping: filtered) { rec -> Int in
            year(of: rec.parsedDate ?? .distantPast)
        }
        return grouped.keys.sorted().map { y in
            let recs = grouped[y] ?? []
            let avg  = recs.map { $0.probability_percent }.reduce(0, +) / Double(recs.count)
            return (y, avg, recs)
        }
    }

    // Labels / briefings
    var cityLabel: String { city.rawValue }
    var monthLabel: String {
        var c = DateComponents(); c.year = 2020; c.month = selectedMonth; c.day = 1
        let d = gmtCal.date(from: c) ?? Date()
        let f = DateFormatter(); f.setLocalizedDateFormatFromTemplate("MMMM"); f.timeZone = .gmt
        return f.string(from: d)
    }
    var selectedDateLabel: String {
        let f = DateFormatter(); f.setLocalizedDateFormatFromTemplate("MMM d, yyyy"); f.timeZone = .gmt
        return f.string(from: selectedDate)
    }
    var selectedMonthOfSelectedDateLabel: String {
        let f = DateFormatter(); f.setLocalizedDateFormatFromTemplate("MMMM"); f.timeZone = .gmt
        return f.string(from: selectedDate)
    }

    var monthModeBriefingText: String? {
        let recs = monthModeChartSeries
        guard !recs.isEmpty, let cond = monthModeAutoCondition else { return nil }
        let p = recs.map { $0.probability_percent }
        let h = recs.map { $0.historical_rate_percent }
        let u = recs.map { $0.confidence_upper_percent }
        return """
        \(cityLabel) · \(monthLabel): \(cond) averages \(Int(round(avg(p))))% (range \(Int(round(minOrZero(p))))–\(Int(round(maxOrZero(p)))))%. \
        Historical rate averages \(Int(round(avg(h))))% (range \(Int(round(minOrZero(h))))–\(Int(round(maxOrZero(h)))))%. \
        Confidence upper bound averages \(Int(round(avg(u))))% (range \(Int(round(minOrZero(u))))–\(Int(round(maxOrZero(u)))))%.
        """
    }

    var dayModeBriefingInRangeText: String? {
        guard let cond = dayModeAutoConditionInRange,
              let chosen = dayExactRecordsInRange.first(where: { $0.conditionText == cond }) else { return nil }
        return "\(cityLabel) · \(selectedDateLabel): top condition \(cond) with \(Int(round(chosen.probability_percent)))%. " +
               "Historical rate: \(Int(round(chosen.historical_rate_percent)))%. " +
               "Confidence upper: \(Int(round(chosen.confidence_upper_percent)))%."
    }

    var dayModeMonthProgressionBriefingText: String? {
        let perYear = monthProgressionPerYear
        guard !perYear.isEmpty, let cond = autoConditionForMonthProgression else { return nil }
        let p = perYear.map { $0.avgProbability }
        return "\(cityLabel) · No exact data for \(selectedDateLabel). \(selectedMonthOfSelectedDateLabel) progression: \(cond) averages \(Int(round(avg(p))))% across years (range \(Int(round(minOrZero(p))))–\(Int(round(maxOrZero(p)))))%."
    }

    // MARK: ---- Event-safe APIs (ignore address gate; use event’s city) ----
    private var RAIN_CONDITION_KEYWORD: String { "wet" }

    func rainProbability(for date: Date, in city: City) -> Double? {
        let cityData = data(for: city)
        let y = gmtCal.component(.year, from: date)
        let minYr = gmtCal.component(.year, from: hardMin)
        let maxYr = gmtCal.component(.year, from: hardMax)
        let dateStr = isoDateFormatter.string(from: date)

        if (minYr...maxYr).contains(y),
           let rec = cityData.first(where: { $0.date == dateStr && $0.condition.lowercased().contains(RAIN_CONDITION_KEYWORD) }) {
            return rec.probability_percent
        }

        let m = gmtCal.component(.month, from: date)
        let monthWet = cityData.filter { rec in
            guard let d = rec.parsedDate else { return false }
            return month(of: d) == m && d >= hardMin && d <= hardMax &&
                   rec.condition.lowercased().contains(RAIN_CONDITION_KEYWORD)
        }
        guard !monthWet.isEmpty else { return nil }
        return monthWet.map(\.probability_percent).reduce(0, +) / Double(monthWet.count)
    }

    func monthlyRainTrend(forMonth targetMonth: Int, in city: City)
    -> [(year: Int, avgProbability: Double)] {
        let cityData = data(for: city)
        let wet = cityData.filter { rec in
            guard let d = rec.parsedDate else { return false }
            return self.month(of: d) == targetMonth
                && d >= hardMin && d <= hardMax
                && rec.condition.lowercased().contains(RAIN_CONDITION_KEYWORD)
        }
        let groups = Dictionary(grouping: wet) { self.year(of: $0.parsedDate ?? .distantPast) }
        return groups.keys.sorted().map { y in
            let arr = groups[y] ?? []
            let avg = arr.map(\.probability_percent).reduce(0, +) / Double(arr.count)
            return (y, avg)
        }
    }

    // Normalize any date to a fixed reference year so charts can compare across years
    func normalizedDateForChart(from date: Date, referenceYear: Int = 2020) -> Date {
        var comps = DateComponents()
        comps.year = referenceYear
        comps.month = gmtCal.component(.month, from: date)
        comps.day = gmtCal.component(.day, from: date)
        return gmtCal.date(from: comps) ?? date
    }

    // Helpers
    private func topCondition(byAverageProbability records: [WeatherRecord]) -> String? {
        guard !records.isEmpty else { return nil }
        let grouped = Dictionary(grouping: records, by: { $0.conditionText })
        let scored  = grouped.mapValues { $0.map(\.probability_percent).reduce(0, +) / Double($0.count) }
        return scored.max(by: { $0.value < $1.value })?.key
    }
    private func avg(_ a: [Double]) -> Double { a.isEmpty ? 0 : a.reduce(0, +) / Double(a.count) }
    private func minOrZero(_ a: [Double]) -> Double { a.min() ?? 0 }
    private func maxOrZero(_ a: [Double]) -> Double { a.max() ?? 0 }
}

