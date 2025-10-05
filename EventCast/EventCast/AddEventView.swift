//
//  AddEventView.swift
//  ArizonaCast
//
//  Created by Oliver Park on 10/5/25.
//

import SwiftUI
import SwiftData

struct AddEventView: View {
    @Environment(\.dismiss) private var dismiss
    @Environment(\.modelContext) private var context

    @ObservedObject var vm: WeatherViewModel

    @State private var title: String = ""
    @State private var date: Date = Date()
    @State private var address: String = ""
    @State private var saveError: String?
    @State private var showingLocationPicker = false

    var body: some View {
        NavigationStack {
            Form {
                Section(header: Text("Details")) {
                    TextField("Title", text: $title)
                    DatePicker("Date", selection: $date, displayedComponents: .date)
                    TextField("Address (e.g., \(vm.city.addressHint))", text: $address)
                        .onChange(of: address) { _, new in vm.address = new }

                    Button {
                        vm.address = address
                        showingLocationPicker = true
                    } label: {
                        Label("Pick on Map", systemImage: "mappin.circle")
                    }
                }

                if vm.addressMatchesCity {
                    let prob = vm.rainProbability(for: date, in: vm.city) ?? 0   // ✅ fixed
                    Section(header: Text("Preview")) {
                        HStack {
                            Text("Rain chance")
                            Spacer()
                            Text(WeatherEmoji.rainIntensity(prob: prob))
                            Text("\(Int(round(prob)))%").monospacedDigit()
                        }
                    }
                } else {
                    Section {
                        Text("Tip: include “\(vm.city.rawValue)” (e.g., \(vm.city.addressHint)) to calculate local rain chances.")
                            .font(.footnote)
                            .foregroundStyle(.secondary)
                    }
                }

                if let saveError {
                    Section { Text(saveError).foregroundStyle(.red).font(.footnote) }
                }
            }
            .navigationTitle("New Event")
            .toolbar {
                ToolbarItem(placement: .topBarLeading) { Button("Cancel") { dismiss() } }
                ToolbarItem(placement: .topBarTrailing) {
                    Button("Save") { handleSave() }
                        .disabled(title.trimmingCharacters(in: .whitespaces).isEmpty)
                }
            }
        }
        .sheet(isPresented: $showingLocationPicker) {
            LocationPickerView(city: vm.city) { picked in
                address = picked
                vm.address = picked
            }
        }
        .onAppear {
            if address.isEmpty { address = vm.city.addressHint }
            vm.address = address
        }
    }

    private func handleSave() {
        let e = UserEvent(
            title: title.isEmpty ? "Untitled Event" : title,
            date: date,
            address: address,
            city: vm.city      // events store their own city
        )
        context.insert(e)
        do {
            try context.save()
            dismiss()
        } catch {
            saveError = "Could not save event: \(error.localizedDescription)"
        }
    }
}
