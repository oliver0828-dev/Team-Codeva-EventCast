//
//  ContentView.swift
//  ArizonaCast
//
//  Created by Oliver Park on 10/5/25.
//

import SwiftUI
import Charts

struct ContentView: View {
    @ObservedObject var vm: WeatherViewModel
    @State private var showingLocationPicker = false
    @State private var showingSettings = false

    var body: some View {
        NavigationView {
            VStack(spacing: 12) {

                TextField("Enter address (e.g., \(vm.city.addressHint))", text: $vm.address)
                    .textFieldStyle(RoundedBorderTextFieldStyle())
                    .padding(.horizontal)

                HStack {
                    Spacer()
                    Button {
                        showingLocationPicker = true
                    } label: {
                        Label("Pick on Map", systemImage: "mappin.circle")
                    }
                    .font(.footnote)
                }
                .padding(.horizontal)

                Picker("Mode", selection: $vm.mode) {
                    ForEach(SelectionMode.allCases) { m in Text(m.rawValue).tag(m) }
                }
                .pickerStyle(.segmented)
                .padding(.horizontal)

                Group { if vm.mode == .month { monthPicker } else { datePicker } }
                    .padding(.horizontal)

                if !vm.addressMatchesCity {
                    Text("Tip: make sure the address mentions \(vm.city.rawValue).")
                        .font(.subheadline).foregroundStyle(.secondary)
                }

                if vm.addressMatchesCity {
                    if vm.mode == .month { monthChartSection } else { dayChartOrProgressionSection }
                }

                Spacer(minLength: 12)
            }
            .navigationTitle("\(vm.city.rawValue)Cast")
            .toolbar {
                ToolbarItem(placement: .topBarTrailing) {
                    Button { showingSettings = true } label: { Image(systemName: "gear") }
                }
            }
        }
        .sheet(isPresented: $showingLocationPicker) {
            LocationPickerView(city: vm.city) { address in vm.address = address }
        }
        .sheet(isPresented: $showingSettings) { SettingsView() }
    }

    // MARK: Subviews

    private var monthPicker: some View {
        HStack {
            Text("Month:").font(.headline); Spacer()
            Picker("Month", selection: $vm.selectedMonth) {
                ForEach(1...12, id: \.self) { m in Text(monthName(m)).tag(m) }
            }
            .pickerStyle(.menu)
        }
    }

    private var datePicker: some View {
        HStack {
            Text("Date:").font(.headline); Spacer()
            DatePicker("", selection: $vm.selectedDate,
                       in: date("2010-01-01")...date("2030-12-31"),
                       displayedComponents: .date)
            .labelsHidden()
        }
    }

    private var monthChartSection: some View {
        Group {
            if !vm.monthModeChartSeries.isEmpty, let cond = vm.monthModeAutoCondition {
                VStack(alignment: .leading, spacing: 8) {
                    Text("Monthly trend — \(vm.monthLabel) · \(cond)").font(.headline)
                    Chart(vm.monthModeChartSeries, id: \.id) { rec in
                        if let d = rec.parsedDate {
                            let nd = vm.normalizedDateForChart(from: d)
                            LineMark(x: .value("Date", nd),
                                     y: .value("Probability %", rec.probability_percent))
                            PointMark(x: .value("Date", nd),
                                      y: .value("Probability %", rec.probability_percent))
                                .symbolSize(20)
                        }
                    }
                    .frame(height: 220)
                    // ✅ Clamp the X domain to the selected month (prevents spillover)
                    .chartXScale(domain: monthRange(year: 2020, month: vm.selectedMonth))
                    .chartXAxis { AxisMarks(values: .automatic(desiredCount: 6)) }
                    .chartYAxis { AxisMarks(position: .leading) }

                    if let brief = vm.monthModeBriefingText {
                        Text(brief).font(.footnote).foregroundStyle(.secondary).padding(.top, 2)
                    }
                }
                .padding(.horizontal)
            } else {
                Text("No data available for \(vm.monthLabel).")
                    .font(.subheadline).foregroundStyle(.secondary).padding(.horizontal)
            }
        }
    }

    private var dayChartOrProgressionSection: some View {
        Group {
            if !vm.dayExactRecordsInRange.isEmpty {
                VStack(alignment: .leading, spacing: 8) {
                    Text("Conditions on \(vm.selectedDateLabel)").font(.headline)
                    Chart(vm.dayExactRecordsInRange, id: \.id) { rec in
                        BarMark(x: .value("Condition", rec.conditionText),
                                y: .value("Probability %", rec.probability_percent))
                    }
                    .frame(height: 220)
                    .chartYAxis { AxisMarks(position: .leading) }

                    if let brief = vm.dayModeBriefingInRangeText {
                        Text(brief).font(.footnote).foregroundStyle(.secondary).padding(.top, 2)
                    }
                }
                .padding(.horizontal)
            } else if !vm.monthProgressionPerYear.isEmpty, let cond = vm.autoConditionForMonthProgression {
                VStack(alignment: .leading, spacing: 8) {
                    Text("No exact data for \(vm.selectedDateLabel). Showing \(vm.selectedMonthOfSelectedDateLabel) progression — \(cond)")
                        .font(.headline)

                    Chart(vm.monthProgressionPerYear, id: \.year) { point in
                        BarMark(x: .value("Year", point.year),
                                y: .value("Avg Probability %", point.avgProbability))
                    }
                    .frame(height: 220)
                    .chartXAxis {
                        AxisMarks(values: .automatic) { value in
                            if let intVal = value.as(Int.self) {
                                AxisGridLine(); AxisTick(); AxisValueLabel("\(intVal)")
                            }
                        }
                    }
                    .chartYAxis { AxisMarks(position: .leading) }

                    if let brief = vm.dayModeMonthProgressionBriefingText {
                        Text(brief).font(.footnote).foregroundStyle(.secondary).padding(.top, 2)
                    }
                }
                .padding(.horizontal)
            } else {
                Text("No data available for \(vm.selectedDateLabel).")
                    .font(.subheadline).foregroundStyle(.secondary).padding(.horizontal)
            }
        }
    }

    // MARK: Helpers
    private func monthName(_ m: Int) -> String {
        var cal = Calendar(identifier: .iso8601)
        cal.timeZone = TimeZone(secondsFromGMT: 0)!
        var comps = DateComponents(); comps.year = 2020; comps.month = m; comps.day = 1
        let d = cal.date(from: comps) ?? Date()
        let f = DateFormatter(); f.setLocalizedDateFormatFromTemplate("MMMM"); f.timeZone = .gmt
        return f.string(from: d)
    }

    private func monthRange(year: Int, month: Int) -> ClosedRange<Date> {
        var cal = Calendar(identifier: .iso8601); cal.timeZone = TimeZone(secondsFromGMT: 0)!
        var comps = DateComponents(); comps.year = year; comps.month = month; comps.day = 1
        let start = cal.date(from: comps)!
        let end = cal.date(byAdding: DateComponents(month: 1, day: -1), to: start)!
        return start...end
    }

    private func date(_ s: String) -> Date { isoDateFormatter.date(from: s)! }
}

