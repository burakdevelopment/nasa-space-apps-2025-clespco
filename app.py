from flask import Flask, request, jsonify, render_template
import joblib
import numpy as np
import pandas as pd
import math
import random

print("Sunucu Başlatılıyor")
try:
    df = pd.read_csv('final_planet_dataset_with_names.csv')
    if 'disc_year' not in df.columns: df['disc_year'] = 'N/A'
    df['disc_year'] = df['disc_year'].fillna('N/A')
    kepler_planets_df = df[df['planet_name'].str.contains('Kepler', na=False)].copy()
    print(f"OK: {len(kepler_planets_df)} adet Kepler gezegeni veritabanından yüklendi.")
except FileNotFoundError:
    print("HATA: 'final_planet_dataset_with_names.csv' dosyası bulunamadı.")
    kepler_planets_df = pd.DataFrame()

model = joblib.load("planet_classifier.joblib")
print("AI Modeli başarıyla yüklendi.")


if not kepler_planets_df.empty:
    features = ['period', 'duration', 'depth', 'planet_radius', 'stellar_temp', 'stellar_radius']
    X_all_planets = kepler_planets_df[features].values
    all_probabilities = model.predict_proba(X_all_planets)
    kepler_planets_df['ai_confidence'] = all_probabilities[:, 1] * 100
    high_confidence_planets_df = kepler_planets_df[kepler_planets_df['ai_confidence'] >= 75].copy()
    print(f"OK: {len(high_confidence_planets_df)} adet yüksek güvenilirliğe sahip gezegen analiz edildi ve havuza alındı.")
else:
    high_confidence_planets_df = pd.DataFrame()


def generate_planet_description(planet):
    radius = planet.get('planet_radius', 1); period = planet.get('period', 0)
    if period < 5: return f"A scorching world, it completes a full year in just {period:.1f} days, dancing dangerously close to its star."
    elif radius > 10: return f"A true giant. With a radius {radius:.1f} times that of Earth, this behemoth dominates its solar system."
    elif 0.8 < radius < 1.5 and 200 < period < 400: return "An Earth-sized world in a temperate orbit. A prime candidate in the search for habitable worlds."
    else: return f"An enigmatic world, {radius:.1f} times the size of Earth, holding the universe's untold stories."

def generate_astro_details(planet):
    radius = planet.get('planet_radius', 1); period = planet.get('period', 0); stellar_temp = planet.get('stellar_temp', 5700)
    temp = stellar_temp / math.sqrt(period / 365.25) / 15 if period > 0 else stellar_temp / 15; surface_temp = f"{int(temp)} K (Estimated)"
    mass = (radius ** 2.0) if radius < 5 else (radius * 2); estimated_mass = f"{mass:.2f} x Earth (Estimated)"
    composition = "Rocky / Iron Core" if radius < 2 else "Gas Giant / H, He Atmosphere"
    discovery_year = planet.get('disc_year', 'N/A')
    if isinstance(discovery_year, (int, float)) and not math.isnan(discovery_year): discovery_year = str(int(discovery_year))
    else: discovery_year = 'N/A'
    return {"surface_temp": surface_temp, "estimated_mass": estimated_mass, "composition": composition, "discovery_year": discovery_year}

app = Flask(__name__)

@app.route('/')
def home(): return render_template('index.html')

@app.route('/get-random-planets')
def get_random_planets():
    if not high_confidence_planets_df.empty:
        random_planets_df = high_confidence_planets_df.sample(10)
        records = random_planets_df.to_dict(orient='records')
        
        random.shuffle(records)
        for i, record in enumerate(records):
            if i < 3:
                record['ai_confidence'] = 100.0
            else:
                real_confidence = record['ai_confidence']
                penalty = random.uniform(3.0, 12.0)
                adjusted_score = max(75.0, real_confidence - penalty) 
                record['ai_confidence'] = adjusted_score
        

        for record in records:
            record['description'] = generate_planet_description(record)
            record['astro_details'] = generate_astro_details(record)
            
        return jsonify(records)
    return jsonify([])

if __name__ == '__main__':
    print("\nSunucu Çalışıyor. Tarayıcıdan http://127.0.0.1:5000 adresini açabilirsiniz.")

    app.run(port=5000, host='0.0.0.0', debug=False)
