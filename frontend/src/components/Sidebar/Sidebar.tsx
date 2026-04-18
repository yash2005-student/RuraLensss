import { 
  Home, 
  Settings,
  ChevronLeft,
  ChevronRight,
  Map,
  Briefcase,
  Shield,
  CloudSun
} from 'lucide-react';
import { useVillageStore } from '../../store/villageStore';
import { useLanguage } from '../../i18n/LanguageContext';

const menuItems = [
  { id: 'dashboard', icon: Home, labelKey: 'dashboard', label: 'Dashboard' },
  { id: 'map', icon: Map, labelKey: 'mapView', label: '3D Map View' },
  { id: 'schemes', icon: Briefcase, labelKey: 'govSchemes', label: 'Government Schemes' },
  { id: 'aqi-weather', icon: CloudSun, labelKey: 'aqiWeather', label: 'AQI & Weather' },
  { id: 'anonymous-reports', icon: Shield, labelKey: 'citizenReports', label: 'Citizen Reports' },
  { id: 'settings', icon: Settings, labelKey: 'settings', label: 'Settings' },
];

export default function Sidebar() {
  const { activeView, setActiveView, sidebarCollapsed, toggleSidebar, userRole } = useVillageStore();
  const { t } = useLanguage();

  // Filter menu items based on user role
  const getMenuItems = () => {
    if (userRole === 'field_worker') {
      // Field workers see limited menu
      return menuItems.filter(item => 
        ['dashboard', 'map', 'aqi-weather', 'anonymous-reports', 'settings'].includes(item.id)
      );
    }
    if (userRole === 'user') {
      // Citizens see basic menu including anonymous reports
      return menuItems.filter(item => 
        ['dashboard', 'map', 'schemes', 'aqi-weather', 'anonymous-reports', 'settings'].includes(item.id)
      );
    }
    // Admin sees all
    return menuItems;
  };

  const handleMenuClick = (itemId: string) => {
    setActiveView(itemId);
    // On mobile, close sidebar after clicking
    if (window.innerWidth < 768 && !sidebarCollapsed) {
      toggleSidebar();
    }
  };

  return (
    <>
      {/* Mobile Overlay - Only visible when sidebar is open on mobile */}
      {!sidebarCollapsed && (
        <div 
          className="md:hidden fixed inset-0 bg-slate-950/80 backdrop-blur-sm z-20 top-16 bottom-8"
          onClick={toggleSidebar}
          aria-label="Close sidebar"
        />
      )}

      {/* Sidebar - Always render content, just hide/show the container */}
      <aside 
        className={`fixed left-0 bottom-8 bg-slate-900/50 backdrop-blur-md border-r border-white/10 shadow-xl transition-transform duration-300 z-30 ${
          sidebarCollapsed 
            ? '-translate-x-full md:translate-x-0 md:w-16' 
            : 'translate-x-0 w-64'
        }`}
        style={{
          top: 'calc(64px + env(safe-area-inset-top, 0px))'
        }}
      >
        {/* Toggle Button - Desktop Only */}
        <button
          onClick={toggleSidebar}
          className="hidden md:flex absolute -right-3 top-6 w-6 h-6 bg-blue-600 rounded-full items-center justify-center hover:bg-blue-500 transition-colors shadow-lg shadow-blue-500/20 z-50"
          aria-label="Toggle sidebar"
        >
          {sidebarCollapsed ? (
            <ChevronRight size={14} className="text-white" />
          ) : (
            <ChevronLeft size={14} className="text-white" />
          )}
        </button>

        {/* Menu Items - ALWAYS RENDERED */}
        <nav className="p-2 space-y-1 overflow-y-auto h-full">
          {getMenuItems().map((item) => {
            const Icon = item.icon;
            const isActive = activeView === item.id;

            return (
              <button
                key={item.id}
                onClick={() => handleMenuClick(item.id)}
                className={`w-full flex items-center space-x-3 px-3 py-2.5 rounded-lg transition-all duration-200 ${
                  isActive 
                    ? 'bg-blue-600 text-white shadow-lg shadow-blue-500/20' 
                    : 'text-slate-400 hover:bg-white/5 hover:text-white'
                }`}
                title={sidebarCollapsed ? t(item.labelKey, item.label) : ''}
              >
                <Icon size={20} className="flex-shrink-0" />
                {!sidebarCollapsed && (
                  <>
                    <span className="text-sm font-medium flex-1 text-left whitespace-nowrap">
                      {t(item.labelKey, item.label)}
                    </span>
                  </>
                )}
              </button>
            );
          })}
        </nav>
      </aside>
    </>
  );
}
