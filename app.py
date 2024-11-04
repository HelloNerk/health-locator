import os
from flask import Flask, render_template, jsonify, request

app = Flask(__name__)

app.config['TEMPLATES_AUTO_RELOAD'] = True

def generate_map(user_lat, user_lon):
    import pandas as pd
    import folium as fl
    from geopy.distance import great_circle
    from collections import defaultdict
    import heapq

    excel_path = os.path.join(os.path.dirname(__file__), 'assets', 'dataset_actualizado.xlsx')
    df_hospitals = pd.read_excel(excel_path, sheet_name='Hospitales')
    df_connections = pd.read_excel(excel_path, sheet_name='Conexiones')

    map = fl.Map(location=[df_hospitals['norte_sig'].mean(), df_hospitals['este_sig'].mean()], zoom_start=7)

    def dijkstra(adj_list, start_hospital):
        shortest_paths = {hospital: float('inf') for hospital in adj_list}
        shortest_paths[start_hospital] = 0
        priority_queue = [(0, start_hospital)]

        while priority_queue:
            current_distance, current_hospital = heapq.heappop(priority_queue)

            if current_distance > shortest_paths[current_hospital]:
                continue

            for neighbor, distance in adj_list[current_hospital]:
                total_distance = current_distance + distance
                if total_distance < shortest_paths[neighbor]:
                    shortest_paths[neighbor] = total_distance
                    heapq.heappush(priority_queue, (total_distance, neighbor))

        return shortest_paths

    def find_nearest_hospital(user_lat, user_lon, df_hospitals):
        min_distance = float('inf')
        nearest_hospital = None

        for _, hospital in df_hospitals.iterrows():
            hospital_lat = hospital['norte_sig']
            hospital_lon = hospital['este_sig']
            distance = great_circle((user_lat, user_lon), (hospital_lat, hospital_lon)).kilometers
            if distance < min_distance:
                min_distance = distance
                nearest_hospital = hospital['index']

        return nearest_hospital, min_distance

    def create_graph(df_connections, df_hospitals):
        graph = defaultdict(list)
        for _, row in df_connections.iterrows():
            hospital1 = row['Hospital1_Index']
            hospital2 = row['Hospital2_Index']
            distance = row['Distance']
            graph[hospital1].append((hospital2, distance))
            graph[hospital2].append((hospital1, distance))
        return graph

    def kruskal_mst(graph, df_hospitals):
        edges = []
        for hospital in graph:
            for neighbor, weight in graph[hospital]:
                edges.append((weight, hospital, neighbor))
        edges.sort()

        parent = {hospital: hospital for hospital in df_hospitals['index']}

        def find(item):
            if parent[item] != item:
                parent[item] = find(parent[item])
            return parent[item]

        def union(x, y):
            parent[find(x)] = find(y)

        mst = []
        for weight, hospital1, hospital2 in edges:
            if find(hospital1) != find(hospital2):
                union(hospital1, hospital2)
                mst.append((hospital1, hospital2, weight))

        return mst

    graph = create_graph(df_connections, df_hospitals)
    mst = kruskal_mst(graph, df_hospitals)

    for _, hospital in df_hospitals.iterrows():
        fl.Marker(
            [hospital['norte_sig'], hospital['este_sig']],
            popup=f"Hospital {hospital['index']}: {hospital['est_nombre']}",
            icon=fl.Icon(color='red', icon='hospital', prefix='fa')
        ).add_to(map)

    for hospital1, hospital2, _ in mst:
        coord1 = df_hospitals[df_hospitals['index'] == hospital1][['norte_sig', 'este_sig']].values[0]
        coord2 = df_hospitals[df_hospitals['index'] == hospital2][['norte_sig', 'este_sig']].values[0]
        fl.PolyLine([coord1, coord2], color="green", weight=2, opacity=0.8).add_to(map)

    # Usar los parámetros en lugar de input()
    nearest_hospital, distance = find_nearest_hospital(user_lat, user_lon, df_hospitals)

    fl.Marker([user_lat, user_lon], popup='Usuario', icon=fl.Icon(color='blue', icon='user', prefix='fa')).add_to(map)
    nearest_coord = df_hospitals[df_hospitals['index'] == nearest_hospital][['norte_sig', 'este_sig']].values[0]
    fl.PolyLine([[user_lat, user_lon], nearest_coord], color="red", weight=2, opacity=0.8).add_to(map)

    shortest_paths = dijkstra(graph, nearest_hospital)

    sorted_paths = sorted(shortest_paths.items(), key=lambda x: x[1])[:5]
    for hospital, distance in sorted_paths[1:]:
        coord = df_hospitals[df_hospitals['index'] == hospital][['norte_sig', 'este_sig']].values[0]
        fl.PolyLine([nearest_coord, coord], color="blue", weight=2, opacity=0.5).add_to(map)

    # En lugar de guardar el mapa, devolver el HTML
    return map._repr_html_()



@app.route('/')
def home():
    return render_template('index.html')

@app.route('/get_map', methods=['POST'])
def get_map():
    try:
        data = request.json
        user_lat = float(data['latitude'])
        user_lon = float(data['longitude'])
        map_html = generate_map(user_lat, user_lon)
        return jsonify({'map_html': map_html})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port)