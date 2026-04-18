import { useEffect, useMemo, useState } from 'react';
import { CloudSun, Wind, Droplets, Activity, RefreshCw, MapPin } from 'lucide-react';
import { Line } from 'react-chartjs-2';
import 'chart.js/auto';
import { useVillageStore } from '../../store/villageStore';

type WeatherCurrent = {
  temperature_2m?: number;
  relative_humidity_2m?: number;
  wind_speed_10m?: number;
  weather_code?: number;
  time?: string;
};

type AirCurrent = {
  us_aqi?: number;
  pm2_5?: number;
  pm10?: number;
  ozone?: number;
  nitrogen_dioxide?: number;
  sulphur_dioxide?: number;
  carbon_monoxide?: number;
  time?: string;
};

type SeriesPoint = {
  time: string;
  value: number;
};

const REFRESH_MS = 2 * 60 * 1000;

function getAqiBand(aqi?: number) {
  if (aqi === undefined || aqi === null) return { label: 'Unknown', color: 'text-slate-300', bg: 'bg-slate-500/10 border-slate-500/30' };
  if (aqi <= 50) return { label: 'Good', color: 'text-emerald-300', bg: 'bg-emerald-500/10 border-emerald-500/30' };
  if (aqi <= 100) return { label: 'Moderate', color: 'text-yellow-300', bg: 'bg-yellow-500/10 border-yellow-500/30' };
  if (aqi <= 150) return { label: 'Unhealthy (Sensitive)', color: 'text-orange-300', bg: 'bg-orange-500/10 border-orange-500/30' };
  if (aqi <= 200) return { label: 'Unhealthy', color: 'text-red-300', bg: 'bg-red-500/10 border-red-500/30' };
  if (aqi <= 300) return { label: 'Very Unhealthy', color: 'text-fuchsia-300', bg: 'bg-fuchsia-500/10 border-fuchsia-500/30' };
  return { label: 'Hazardous', color: 'text-rose-300', bg: 'bg-rose-500/10 border-rose-500/30' };
}

function weatherCodeLabel(code?: number) {
  if (code === undefined || code === null) return 'Unknown';
  if (code === 0) return 'Clear sky';
  if ([1, 2, 3].includes(code)) return 'Partly cloudy';
  if ([45, 48].includes(code)) return 'Fog';
  if ([51, 53, 55, 56, 57].includes(code)) return 'Drizzle';
  if ([61, 63, 65, 66, 67].includes(code)) return 'Rain';
  if ([71, 73, 75, 77].includes(code)) return 'Snow';
  if ([80, 81, 82].includes(code)) return 'Rain showers';
  if ([85, 86].includes(code)) return 'Snow showers';
  if ([95, 96, 99].includes(code)) return 'Thunderstorm';
  return 'Variable';
}

export default function AqiWeatherView() {
  const { waterTanks, sensors, powerNodes } = useVillageStore();
  const [weather, setWeather] = useState<WeatherCurrent | null>(null);
  const [air, setAir] = useState<AirCurrent | null>(null);
  const [tempSeries, setTempSeries] = useState<SeriesPoint[]>([]);
  const [aqiSeries, setAqiSeries] = useState<SeriesPoint[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [lastUpdated, setLastUpdated] = useState<string | null>(null);

  const coords = useMemo(() => {
    const points: Array<[number, number]> = [];
    waterTanks.forEach((t) => Array.isArray(t.coords) && points.push(t.coords));
    sensors.forEach((s) => Array.isArray(s.coords) && points.push(s.coords));
    powerNodes.forEach((p) => Array.isArray(p.coords) && points.push(p.coords));

    if (points.length === 0) {
      return { lon: 73.8567, lat: 18.5204 }; // Pune fallback
    }

    const lon = points.reduce((acc, c) => acc + c[0], 0) / points.length;
    const lat = points.reduce((acc, c) => acc + c[1], 0) / points.length;
    return { lon, lat };
  }, [waterTanks, sensors, powerNodes]);

  const fetchLiveData = async () => {
    try {
      setError(null);
      const weatherUrl = `https://api.open-meteo.com/v1/forecast?latitude=${coords.lat}&longitude=${coords.lon}&current=temperature_2m,relative_humidity_2m,wind_speed_10m,weather_code&hourly=temperature_2m&past_days=1&forecast_days=1&timezone=auto`;
      const airUrl = `https://air-quality-api.open-meteo.com/v1/air-quality?latitude=${coords.lat}&longitude=${coords.lon}&current=us_aqi,pm2_5,pm10,ozone,nitrogen_dioxide,sulphur_dioxide,carbon_monoxide&hourly=us_aqi&past_days=1&forecast_days=1&timezone=auto`;

      const [weatherRes, airRes] = await Promise.all([fetch(weatherUrl), fetch(airUrl)]);
      if (!weatherRes.ok || !airRes.ok) {
        throw new Error('Unable to fetch live environment data');
      }

      const weatherJson = await weatherRes.json();
      const airJson = await airRes.json();

      setWeather(weatherJson.current || null);
      setAir(airJson.current || null);

      const weatherTimes: string[] = weatherJson?.hourly?.time || [];
      const weatherVals: number[] = weatherJson?.hourly?.temperature_2m || [];
      const weatherSeries = weatherTimes.map((t: string, i: number) => ({
        time: t,
        value: Number(weatherVals[i] ?? 0)
      }));
      setTempSeries(weatherSeries.slice(-24));

      const airTimes: string[] = airJson?.hourly?.time || [];
      const airVals: number[] = airJson?.hourly?.us_aqi || [];
      const airSeries = airTimes.map((t: string, i: number) => ({
        time: t,
        value: Number(airVals[i] ?? 0)
      }));
      setAqiSeries(airSeries.slice(-24));

      setLastUpdated(new Date().toISOString());
    } catch (e: any) {
      setError(e?.message || 'Failed to fetch data');
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    setLoading(true);
    fetchLiveData();
    const timer = setInterval(fetchLiveData, REFRESH_MS);
    return () => clearInterval(timer);
  }, [coords.lat, coords.lon]);

  const aqiBand = getAqiBand(air?.us_aqi);
  const mapFrameUrl = useMemo(() => {
    const lonDelta = 0.03;
    const latDelta = 0.02;
    const left = coords.lon - lonDelta;
    const right = coords.lon + lonDelta;
    const top = coords.lat + latDelta;
    const bottom = coords.lat - latDelta;
    return `https://www.openstreetmap.org/export/embed.html?bbox=${left}%2C${bottom}%2C${right}%2C${top}&layer=mapnik&marker=${coords.lat}%2C${coords.lon}`;
  }, [coords.lat, coords.lon]);

  const chartOptions = useMemo(
    () => ({
      responsive: true,
      maintainAspectRatio: false,
      plugins: {
        legend: {
          labels: {
            color: '#cbd5e1'
          }
        }
      },
      scales: {
        x: {
          ticks: { color: '#94a3b8', maxTicksLimit: 8 },
          grid: { color: 'rgba(148, 163, 184, 0.18)' }
        },
        y: {
          ticks: { color: '#94a3b8' },
          grid: { color: 'rgba(148, 163, 184, 0.18)' }
        }
      }
    }),
    []
  );

  const tempChartData = useMemo(
    () => ({
      labels: tempSeries.map((p) => new Date(p.time).toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })),
      datasets: [
        {
          label: 'Temperature (°C)',
          data: tempSeries.map((p) => p.value),
          borderColor: '#38bdf8',
          backgroundColor: 'rgba(56, 189, 248, 0.2)',
          pointRadius: 2,
          tension: 0.35,
          fill: true
        }
      ]
    }),
    [tempSeries]
  );

  const aqiChartData = useMemo(
    () => ({
      labels: aqiSeries.map((p) => new Date(p.time).toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })),
      datasets: [
        {
          label: 'US AQI',
          data: aqiSeries.map((p) => p.value),
          borderColor: '#34d399',
          backgroundColor: 'rgba(52, 211, 153, 0.2)',
          pointRadius: 2,
          tension: 0.35,
          fill: true
        }
      ]
    }),
    [aqiSeries]
  );

  return (
    <div className="h-full overflow-y-auto p-6 bg-slate-950 text-slate-200">
      <div className="max-w-6xl mx-auto space-y-6">
        <div className="flex items-center justify-between flex-wrap gap-3">
          <div>
            <h1 className="text-2xl font-bold text-white">AQI & Temperature</h1>
            <p className="text-slate-400 text-sm flex items-center gap-1 mt-1">
              <MapPin size={14} />
              Live data for {coords.lat.toFixed(4)}, {coords.lon.toFixed(4)} (Open-Meteo)
            </p>
          </div>
          <button
            onClick={fetchLiveData}
            className="inline-flex items-center gap-2 px-3 py-2 rounded-lg bg-slate-800 border border-white/10 text-slate-200 hover:bg-slate-700 transition-colors"
          >
            <RefreshCw size={16} />
            Refresh
          </button>
        </div>

        {error && (
          <div className="rounded-xl border border-red-500/30 bg-red-500/10 text-red-300 px-4 py-3 text-sm">
            {error}
          </div>
        )}

        <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
          <div className="rounded-2xl border border-white/10 bg-slate-900/60 p-5">
            <div className="flex items-center gap-2 text-slate-300 mb-3">
              <CloudSun size={18} className="text-sky-400" />
              <span className="font-semibold">Temperature</span>
            </div>
            <div className="text-4xl font-bold text-white">
              {loading ? '--' : `${Math.round(weather?.temperature_2m ?? 0)}°C`}
            </div>
            <p className="text-slate-400 mt-2">{weatherCodeLabel(weather?.weather_code)}</p>
          </div>

          <div className={`rounded-2xl border p-5 ${aqiBand.bg}`}>
            <div className="flex items-center gap-2 text-slate-300 mb-3">
              <Activity size={18} className="text-emerald-400" />
              <span className="font-semibold">Air Quality (US AQI)</span>
            </div>
            <div className="text-4xl font-bold text-white">
              {loading ? '--' : `${Math.round(air?.us_aqi ?? 0)}`}
            </div>
            <p className={`mt-2 font-medium ${aqiBand.color}`}>{aqiBand.label}</p>
          </div>
        </div>

        <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
          <div className="rounded-xl border border-white/10 bg-slate-900/50 p-4">
            <p className="text-slate-400 text-sm flex items-center gap-2"><Droplets size={14} />Humidity</p>
            <p className="text-xl font-semibold text-white mt-1">{loading ? '--' : `${Math.round(weather?.relative_humidity_2m ?? 0)}%`}</p>
          </div>
          <div className="rounded-xl border border-white/10 bg-slate-900/50 p-4">
            <p className="text-slate-400 text-sm flex items-center gap-2"><Wind size={14} />Wind</p>
            <p className="text-xl font-semibold text-white mt-1">{loading ? '--' : `${Math.round(weather?.wind_speed_10m ?? 0)} km/h`}</p>
          </div>
          <div className="rounded-xl border border-white/10 bg-slate-900/50 p-4">
            <p className="text-slate-400 text-sm">PM2.5 / PM10</p>
            <p className="text-xl font-semibold text-white mt-1">
              {loading ? '--' : `${Math.round(air?.pm2_5 ?? 0)} / ${Math.round(air?.pm10 ?? 0)} µg/m³`}
            </p>
          </div>
        </div>

        <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
          <div className="rounded-xl border border-white/10 bg-slate-900/50 p-4">
            <p className="text-slate-400 text-sm">Ozone (O₃)</p>
            <p className="text-lg font-semibold text-white mt-1">{loading ? '--' : `${Math.round(air?.ozone ?? 0)} µg/m³`}</p>
          </div>
          <div className="rounded-xl border border-white/10 bg-slate-900/50 p-4">
            <p className="text-slate-400 text-sm">NO₂</p>
            <p className="text-lg font-semibold text-white mt-1">{loading ? '--' : `${Math.round(air?.nitrogen_dioxide ?? 0)} µg/m³`}</p>
          </div>
          <div className="rounded-xl border border-white/10 bg-slate-900/50 p-4">
            <p className="text-slate-400 text-sm">SO₂ / CO</p>
            <p className="text-lg font-semibold text-white mt-1">
              {loading ? '--' : `${Math.round(air?.sulphur_dioxide ?? 0)} / ${Math.round(air?.carbon_monoxide ?? 0)} µg/m³`}
            </p>
          </div>
        </div>

        <div className="grid grid-cols-1 xl:grid-cols-2 gap-4">
          <div className="rounded-2xl border border-white/10 bg-slate-900/60 p-4">
            <div className="flex items-center gap-2 text-slate-300 mb-3">
              <MapPin size={18} className="text-blue-400" />
              <span className="font-semibold">Location Map</span>
            </div>
            <div className="w-full h-[320px] rounded-xl overflow-hidden border border-white/10 bg-slate-800">
              <iframe
                title="AQI Weather Location Map"
                src={mapFrameUrl}
                className="w-full h-full"
                loading="lazy"
              />
            </div>
          </div>

          <div className="rounded-2xl border border-white/10 bg-slate-900/60 p-4 space-y-4">
            <div>
              <p className="text-slate-300 font-semibold mb-2">Temperature Trend (Last 24 points)</p>
              <div className="h-36">
                <Line data={tempChartData} options={chartOptions} />
              </div>
            </div>
            <div>
              <p className="text-slate-300 font-semibold mb-2">AQI Trend (Last 24 points)</p>
              <div className="h-36">
                <Line data={aqiChartData} options={chartOptions} />
              </div>
            </div>
          </div>
        </div>

        <p className="text-xs text-slate-500">
          {lastUpdated ? `Last updated: ${new Date(lastUpdated).toLocaleString()}` : 'Waiting for first data fetch...'}
        </p>
      </div>
    </div>
  );
}
