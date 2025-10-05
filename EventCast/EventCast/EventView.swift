//
//  EventView.swift
//  ArizonaCast
//
//  Created by Oliver Park on 10/5/25.
//

import SwiftUI
import SwiftData

struct EventsView: View {
    @Environment(\.modelContext) private var context
    @Query(sort: \UserEvent.date, order: .forward) private var events: [UserEvent]
    @ObservedObject var vm: WeatherViewModel
    @State private var showAdd = false

    var body: some View {
        NavigationStack {
            List {
                Section(header: Text("Events (\(events.count))")) {
                    ForEach(events) { ev in
                        NavigationLink {
                            EventDetailView(event: ev, vm: vm)
                        } label: {
                            EventRow(event: ev, vm: vm)
                        }
                    }
                    .onDelete { idx in
                        idx.map { events[$0] }.forEach(context.delete)
                        try? context.save()
                    }
                }
            }
            .navigationTitle("My Events")
            .toolbar {
                ToolbarItem(placement: .topBarTrailing) {
                    Button { showAdd = true } label: {
                        Label("Add", systemImage: "plus.circle.fill")
                    }
                }
            }
            .sheet(isPresented: $showAdd) { AddEventView(vm: vm) }
            .task {
                // Debug: print the store path so you can verify one container
                if let url = context.container.configurations.first?.url {
                    print("SwiftData store URL:", url.path)
                }
            }
        }
    }
}

private struct EventRow: View {
    let event: UserEvent
    @ObservedObject var vm: WeatherViewModel
    var body: some View {
        let prob = vm.rainProbability(for: event.date, in: event.city) ?? 0
        let feeling = WeatherEmoji.rainFeelingLevel(prob: prob)
        HStack {
            Text(WeatherEmoji.rainIntensity(prob: prob)).font(.title2)
            VStack(alignment: .leading) {
                Text(event.title).font(.headline)
                Text("\(event.city.rawValue) • \(event.date, style: .date)")
                    .font(.subheadline).foregroundStyle(.secondary)
                Text(feeling).font(.footnote).foregroundStyle(.secondary)
            }
            Spacer()
            Text("\(Int(round(prob)))%").monospacedDigit().font(.headline)
        }
        .padding(.vertical, 4)
    }
}
