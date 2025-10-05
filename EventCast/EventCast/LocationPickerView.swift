import SwiftUI
import MapKit
import Combine

struct LocationPickerView: View {
    @Environment(\.dismiss) private var dismiss

    let city: City
    var onPick: (String) -> Void

    @StateObject private var search: CityLocalSearch
    @State private var region: MKCoordinateRegion
    @State private var selectedItem: MKMapItem?
    @State private var selectedPlace: IdentifiablePlace?
    @State private var showNonCityAlert = false

    init(city: City, onPick: @escaping (String) -> Void) {
        self.city = city
        self.onPick = onPick
        _region = State(initialValue: city.mapRegion)
        _search = StateObject(wrappedValue: CityLocalSearch(city: city))
    }

    var body: some View {
        NavigationStack {
            VStack(spacing: 0) {
                Map(coordinateRegion: $region, annotationItems: selectedPlace.map { [$0] } ?? []) { place in
                    MapMarker(coordinate: place.coordinate)
                }
                .frame(height: 260)
                .accessibilityLabel("Map preview of selected location")

                List {
                    Section {
                        TextField("Search near: \(city.addressHint)", text: $search.query)
                            .textInputAutocapitalization(.words)
                            .disableAutocorrection(true)
                    }

                    Section(header: Text("Results in \(city.rawValue)")) {
                        if search.results.isEmpty && !search.query.isEmpty {
                            Text("No matches in \(city.rawValue).")
                                .foregroundStyle(.secondary)
                        }
                        ForEach(search.results, id: \.self) { completion in
                            Button(action: { resolve(completion) }) {
                                VStack(alignment: .leading, spacing: 2) {
                                    Text(completion.title)
                                    if !completion.subtitle.isEmpty {
                                        Text(completion.subtitle)
                                            .font(.footnote)
                                            .foregroundStyle(.secondary)
                                    }
                                }
                            }
                        }
                    }
                }
                .listStyle(.insetGrouped)
            }
            .navigationTitle("Pick Location")
            .toolbar {
                ToolbarItem(placement: .topBarLeading) {
                    Button("Cancel") { dismiss() }
                }
                ToolbarItem(placement: .topBarTrailing) {
                    Button("Use") { commitSelection() }
                        .disabled(selectedItem == nil)
                }
                ToolbarItem(placement: .topBarTrailing) {
                    Button {
                        region = city.mapRegion
                        selectedPlace = IdentifiablePlace(coordinate: city.coordinate, name: city.rawValue)
                        selectedItem = nil
                    } label: {
                        Image(systemName: "scope")
                    }
                }
            }
            .alert("Please pick a location in \(city.rawValue)", isPresented: $showNonCityAlert) {
                Button("OK", role: .cancel) { }
            }
            .onAppear {
                region = city.mapRegion
                if selectedPlace == nil {
                    selectedPlace = IdentifiablePlace(coordinate: city.coordinate,
                                                      name: city.rawValue)
                }
            }
        }
    }

    private func resolve(_ completion: MKLocalSearchCompletion) {
        let request = MKLocalSearch.Request(completion: completion)
        request.region = city.mapRegion
        request.naturalLanguageQuery = completion.title + ", " + city.rawValue

        let search = MKLocalSearch(request: request)
        search.start { response, error in
            guard error == nil, let item = response?.mapItems.first else { return }
            let joined = joinedAddress(from: item.placemark, title: item.name)
            if city.matches(address: joined) {
                withAnimation {
                    selectedItem = item
                    selectedPlace = IdentifiablePlace(coordinate: item.placemark.coordinate,
                                                      name: item.name ?? "Selected Location")
                    region.center = item.placemark.coordinate
                    region.span = MKCoordinateSpan(latitudeDelta: 0.2, longitudeDelta: 0.2)
                }
            } else {
                showNonCityAlert = true
            }
        }
    }

    private func commitSelection() {
        guard let item = selectedItem else { return }
        let address = joinedAddress(from: item.placemark, title: item.name)
        onPick(address)
        dismiss()
    }

    private func joinedAddress(from placemark: MKPlacemark, title: String?) -> String {
        let parts: [String?] = [
            title,
            placemark.locality,
            placemark.administrativeArea,
            placemark.country
        ]
        return parts.compactMap { $0?.trimmingCharacters(in: .whitespacesAndNewlines) }
                    .filter { !$0.isEmpty }
                    .joined(separator: ", ")
    }
}

private struct IdentifiablePlace: Identifiable {
    let id = UUID()
    let coordinate: CLLocationCoordinate2D
    let name: String
}

// Local search bounded to a specific city
final class CityLocalSearch: NSObject, ObservableObject, MKLocalSearchCompleterDelegate {
    @Published var query: String = "" {
        didSet { completer.queryFragment = query }
    }
    @Published var results: [MKLocalSearchCompletion] = []

    let city: City
    let completer: MKLocalSearchCompleter = MKLocalSearchCompleter()

    init(city: City) {
        self.city = city
        super.init()
        completer.delegate = self
        completer.resultTypes = .address
        completer.region = city.mapRegion
    }

    func completerDidUpdateResults(_ completer: MKLocalSearchCompleter) {
        let filtered = completer.results.filter { res in
            let t = (res.title + " " + res.subtitle).lowercased()
            return city.matches(address: t)
        }
        self.results = filtered
    }

    func completer(_ completer: MKLocalSearchCompleter, didFailWithError error: Error) {
        self.results = []
    }
}
