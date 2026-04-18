import { 
  Briefcase, 
  TrendingUp, 
  AlertTriangle,
  DollarSign,
  Users,
  Star,
  ArrowRight,
  CheckCircle,
  Clock,
  Target,
  Activity,
  Zap,
  Sparkles,
  MessageSquare
} from 'lucide-react';
import { useState } from 'react';
import { useVillageStore } from '../../store/villageStore';
import RagQueryModal from '../Rag/RagQueryModal';
import type { Citation } from '../../hooks/useRagQuery';
import { useLanguage } from '../../i18n/LanguageContext';

export default function AdminDashboard() {
  const schemes = useVillageStore((state) => state.schemes);
  const setActiveView = useVillageStore((state) => state.setActiveView);
  const [showRagModal, setShowRagModal] = useState(false);
  const { lang } = useLanguage();
  const hi = lang === 'hi';
  const tx = (en: string, hiText: string) => (hi ? hiText : en);

  // Calculate KPIs
  const totalSchemes = schemes.length;
  const onTrackSchemes = schemes.filter(s => s.status === 'on-track' || s.status === 'completed').length;
  const delayedSchemes = schemes.filter(s => s.status === 'delayed').length;
  const discrepantSchemes = schemes.filter(s => s.status === 'discrepant').length;
  const completedSchemes = schemes.filter(s => s.status === 'completed').length;
  const totalBudget = schemes.reduce((sum, s) => sum + s.totalBudget, 0);
  const budgetUtilized = schemes.reduce((sum, s) => sum + s.budgetUtilized, 0);
  const avgProgress = schemes.length > 0 ? Math.round(schemes.reduce((sum, s) => sum + s.overallProgress, 0) / schemes.length) : 0;
  const totalFeedback = schemes.reduce((sum, s) => sum + s.feedbackCount, 0);
  const avgRating = schemes.length > 0 ? (schemes.reduce((sum, s) => sum + s.citizenRating, 0) / schemes.length).toFixed(1) : '0.0';

  // Budget by category
  const budgetByCategory = schemes.reduce((acc: Record<string, { allocated: number; used: number }>, scheme) => {
    if (!acc[scheme.category]) {
      acc[scheme.category] = { allocated: 0, used: 0 };
    }
    acc[scheme.category].allocated += scheme.totalBudget;
    acc[scheme.category].used += scheme.budgetUtilized;
    return acc;
  }, {});

  // Schemes needing attention
  const needsAttention = schemes
    .filter(s => s.status === 'delayed' || s.status === 'discrepant')
    .slice(0, 3);

  // Only show first 2 schemes in dashboard for preview
  const displayedSchemes = schemes.slice(0, 2);

  const getStatusColor = (status: string) => {
    switch (status) {
      case 'on-track': return 'bg-green-100 text-green-800 border-green-200';
      case 'delayed': return 'bg-yellow-100 text-yellow-800 border-yellow-200';
      case 'discrepant': return 'bg-red-100 text-red-800 border-red-200';
      case 'completed': return 'bg-blue-100 text-blue-800 border-blue-200';
      default: return 'bg-gray-100 text-gray-800 border-gray-200';
    }
  };

  // Category mapping for icons
  const categoryIcons: Record<string, string> = {
    'Sanitation': '🧹',
    'Water Supply': '💧',
    'Housing': '🏠',
    'Employment': '👷',
    'Power': '⚡'
  };

  const categoryColors: Record<string, string> = {
    'Sanitation': 'bg-blue-500',
    'Water Supply': 'bg-cyan-500',
    'Housing': 'bg-orange-500',
    'Employment': 'bg-purple-500',
    'Power': 'bg-yellow-500'
  };

  // Status badge component
  const StatusBadge = ({ status }: { status: string }) => {
    return (
      <span className={`px-2 py-1 rounded-full text-xs font-medium border ${getStatusColor(status)}`}>
        {status.replace('-', ' ').toUpperCase()}
      </span>
    );
  };

  return (
    <div className="h-full overflow-y-auto p-3 md:p-6 space-y-4 md:space-y-6 bg-transparent">
      {/* Header */}
      <div className="mb-4 md:mb-6 flex items-start justify-between gap-4">
        <div>
          <h1 className="text-2xl md:text-3xl font-bold text-white mb-1 md:mb-2">{tx('Admin Dashboard', 'एडमिन डैशबोर्ड')}</h1>
          <p className="text-sm md:text-base text-slate-400">{tx('Real-time overview • Updated just now', 'रियल-टाइम ओवरव्यू • अभी अपडेट हुआ')}</p>
        </div>
        <button
          onClick={() => setShowRagModal(true)}
          className="px-4 py-2 bg-gradient-to-r from-blue-600 to-cyan-600 text-white rounded-lg font-medium hover:from-blue-500 hover:to-cyan-500 transition-all shadow-lg shadow-blue-500/20 flex items-center space-x-2"
        >
          <Sparkles size={18} />
          <span className="hidden sm:inline">{tx('Ask AI', 'एआई से पूछें')}</span>
        </button>
      </div>

      {/* Main Dashboard Grid */}
      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        
        {/* Left Column (2/3) - Main Metrics & Charts */}
        <div className="lg:col-span-2 space-y-6">
          
          {/* KPI Cards Row */}
          <div className="grid grid-cols-2 sm:grid-cols-4 gap-4">
            <div className="bg-slate-900/40 border border-white/10 rounded-xl p-4 flex flex-col justify-between h-32 relative overflow-hidden group">
              <div className="absolute right-0 top-0 p-3 opacity-10 group-hover:opacity-20 transition-opacity">
                <Briefcase size={48} />
              </div>
              <div className="text-slate-400 text-xs font-medium uppercase tracking-wider">{tx('Total Projects', 'कुल परियोजनाएं')}</div>
              <div>
                <div className="text-3xl font-bold text-white">{totalSchemes}</div>
                <div className="text-xs text-emerald-400 mt-1 flex items-center gap-1">
                  <CheckCircle size={12} /> {completedSchemes} {tx('Completed', 'पूर्ण')}
                </div>
              </div>
            </div>

            <div className="bg-slate-900/40 border border-white/10 rounded-xl p-4 flex flex-col justify-between h-32 relative overflow-hidden group">
              <div className="absolute right-0 top-0 p-3 opacity-10 group-hover:opacity-20 transition-opacity">
                <Target size={48} />
              </div>
              <div className="text-slate-400 text-xs font-medium uppercase tracking-wider">{tx('On Track', 'समय पर')}</div>
              <div>
                <div className="text-3xl font-bold text-white">{onTrackSchemes}</div>
                <div className="text-xs text-emerald-400 mt-1 flex items-center gap-1">
                  <TrendingUp size={12} /> {Math.round((onTrackSchemes / totalSchemes) * 100)}% {tx('Rate', 'दर')}
                </div>
              </div>
            </div>

            <div className="bg-slate-900/40 border border-white/10 rounded-xl p-4 flex flex-col justify-between h-32 relative overflow-hidden group">
              <div className="absolute right-0 top-0 p-3 opacity-10 group-hover:opacity-20 transition-opacity">
                <AlertTriangle size={48} />
              </div>
              <div className="text-slate-400 text-xs font-medium uppercase tracking-wider">{tx('Critical', 'गंभीर')}</div>
              <div>
                <div className="text-3xl font-bold text-white">{discrepantSchemes}</div>
                <div className="text-xs text-rose-400 mt-1 flex items-center gap-1">
                  <Clock size={12} /> {delayedSchemes} {tx('Delayed', 'विलंबित')}
                </div>
              </div>
            </div>

            <div className="bg-slate-900/40 border border-white/10 rounded-xl p-4 flex flex-col justify-between h-32 relative overflow-hidden group">
              <div className="absolute right-0 top-0 p-3 opacity-10 group-hover:opacity-20 transition-opacity">
                <Activity size={48} />
              </div>
              <div className="text-slate-400 text-xs font-medium uppercase tracking-wider">{tx('Avg Progress', 'औसत प्रगति')}</div>
              <div>
                <div className="text-3xl font-bold text-white">{avgProgress}%</div>
                <div className="text-xs text-blue-400 mt-1 flex items-center gap-1">
                  <TrendingUp size={12} /> {tx('+5% MoM', '+5% मासिक')}
                </div>
              </div>
            </div>
          </div>

          {/* Financial Overview Section */}
          <div className="bg-slate-900/40 border border-white/10 rounded-xl p-6">
            <div className="flex items-center justify-between mb-6">
              <h3 className="text-lg font-bold text-white flex items-center gap-2">
                <DollarSign size={20} className="text-emerald-400" />
                {tx('Financial Overview', 'वित्तीय अवलोकन')}
              </h3>
              <div className="text-sm text-slate-400">
                {tx('Total Budget:', 'कुल बजट:')} <span className="text-white font-mono">₹{(totalBudget / 10000000).toFixed(2)} Cr</span>
              </div>
            </div>

            {/* Budget Progress Bar */}
            <div className="mb-8">
              <div className="flex justify-between text-sm mb-2">
                <span className="text-slate-400">{tx('Utilization', 'उपयोग')}</span>
                <span className="text-white font-bold">{Math.round((budgetUtilized / totalBudget) * 100)}%</span>
              </div>
              <div className="h-4 bg-slate-800 rounded-full overflow-hidden relative">
                <div 
                  className="h-full bg-gradient-to-r from-emerald-600 to-emerald-400 rounded-full"
                  style={{ width: `${Math.round((budgetUtilized / totalBudget) * 100)}%` }}
                />
                {/* Markers */}
                <div className="absolute top-0 bottom-0 w-0.5 bg-white/20 left-[25%]"></div>
                <div className="absolute top-0 bottom-0 w-0.5 bg-white/20 left-[50%]"></div>
                <div className="absolute top-0 bottom-0 w-0.5 bg-white/20 left-[75%]"></div>
              </div>
              <div className="flex justify-between text-xs text-slate-500 mt-1">
                <span>0%</span>
                <span>25%</span>
                <span>50%</span>
                <span>75%</span>
                <span>100%</span>
              </div>
            </div>

            {/* Category Budget Breakdown */}
            <div className="grid grid-cols-1 sm:grid-cols-2 gap-4">
              {Object.entries(budgetByCategory).slice(0, 4).map(([category, data]) => (
                <div key={category} className="bg-slate-800/30 rounded-lg p-3 border border-white/5">
                  <div className="flex justify-between items-center mb-2">
                    <div className="flex items-center gap-2">
                      <span className="text-lg">{categoryIcons[category]}</span>
                      <span className="text-sm font-medium text-slate-300">{category}</span>
                    </div>
                    <span className="text-xs font-mono text-slate-400">₹{(data.allocated / 100000).toFixed(0)}L</span>
                  </div>
                  <div className="h-1.5 bg-slate-700 rounded-full overflow-hidden">
                    <div 
                      className={`h-full ${categoryColors[category] || 'bg-blue-500'}`}
                      style={{ width: `${(data.used / data.allocated) * 100}%` }}
                    />
                  </div>
                </div>
              ))}
            </div>
          </div>

          {/* Recent Schemes Table Preview */}
          <div className="bg-slate-900/40 border border-white/10 rounded-xl overflow-hidden">
            <div className="p-4 border-b border-white/10 flex justify-between items-center">
              <h3 className="font-bold text-white">{tx('Recent Projects', 'हाल की परियोजनाएं')}</h3>
              <button 
                onClick={() => setActiveView('schemes')}
                className="text-xs text-purple-400 hover:text-purple-300 flex items-center gap-1"
              >
                {tx('View All', 'सभी देखें')} <ArrowRight size={12} />
              </button>
            </div>
            <div className="divide-y divide-white/5">
              {displayedSchemes.map((scheme) => (
                <div key={scheme.id} className="p-4 hover:bg-white/5 transition-colors flex items-center justify-between group cursor-pointer" onClick={() => setActiveView('schemes')}>
                  <div className="flex items-center gap-3">
                    <div className="w-10 h-10 rounded-lg bg-slate-800 flex items-center justify-center text-xl border border-white/5">
                      {categoryIcons[scheme.category]}
                    </div>
                    <div>
                      <div className="font-medium text-white group-hover:text-purple-400 transition-colors">{scheme.name}</div>
                      <div className="text-xs text-slate-500">{scheme.village} • ID: {scheme.id}</div>
                    </div>
                  </div>
                  <div className="flex items-center gap-4">
                    <div className="text-right hidden sm:block">
                      <div className="text-xs text-slate-400">{tx('Progress', 'प्रगति')}</div>
                      <div className="font-bold text-white">{scheme.overallProgress}%</div>
                    </div>
                    <StatusBadge status={scheme.status} />
                  </div>
                </div>
              ))}
            </div>
          </div>

        </div>

        {/* Right Column (1/3) - Alerts & Actions */}
        <div className="space-y-6">
          
          {/* Priority Alerts Panel */}
          <div className="bg-slate-900/40 border border-white/10 rounded-xl p-5 h-auto">
            <div className="flex items-center justify-between mb-4">
              <h3 className="font-bold text-white flex items-center gap-2">
                <AlertTriangle size={18} className="text-rose-500" />
                {tx('Critical Alerts', 'गंभीर अलर्ट')}
              </h3>
              <span className="bg-rose-500/20 text-rose-400 text-xs font-bold px-2 py-1 rounded-full">
                {needsAttention.length}
              </span>
            </div>
            
            <div className="space-y-3">
              {needsAttention.length > 0 ? (
                needsAttention.map((scheme) => (
                  <div key={scheme.id} className="bg-rose-500/5 border border-rose-500/10 rounded-lg p-3 hover:bg-rose-500/10 transition-colors cursor-pointer" onClick={() => setActiveView('schemes')}>
                    <div className="flex justify-between items-start mb-1">
                      <span className="text-xs font-bold text-rose-300 uppercase tracking-wider">{tx('Action Required', 'तत्काल कार्रवाई')}</span>
                      <span className="text-[10px] text-slate-500">{new Date(scheme.lastUpdated).toLocaleDateString()}</span>
                    </div>
                    <div className="text-sm font-medium text-white mb-1 line-clamp-1">{scheme.name}</div>
                    <div className="text-xs text-slate-400">
                      {scheme.discrepancies.length > 0 ? `${scheme.discrepancies.length} ${tx('discrepancies detected', 'असंगतियां मिलीं')}` : tx('Project delayed significantly', 'परियोजना में महत्वपूर्ण देरी')}
                    </div>
                  </div>
                ))
              ) : (
                <div className="text-center py-8 text-slate-500">
                  <CheckCircle size={32} className="mx-auto mb-2 text-emerald-500/50" />
                  <p className="text-sm">{tx('All systems operational', 'सभी सिस्टम सामान्य हैं')}</p>
                </div>
              )}
            </div>
          </div>

          {/* Quick Actions Panel */}
          <div className="bg-slate-900/40 border border-white/10 rounded-xl p-5">
            <h3 className="font-bold text-white mb-4 flex items-center gap-2">
              <Zap size={18} className="text-yellow-500" />
              {tx('Quick Actions', 'त्वरित कार्रवाइयां')}
            </h3>
            <div className="grid grid-cols-1 gap-2">
              <button 
                onClick={() => setActiveView('schemes')}
                className="flex items-center gap-3 p-3 rounded-lg bg-slate-800/50 hover:bg-purple-600/20 border border-white/5 hover:border-purple-500/30 transition-all group text-left"
              >
                <div className="p-2 rounded-md bg-purple-500/20 text-purple-400 group-hover:bg-purple-500 group-hover:text-white transition-colors">
                  <Briefcase size={16} />
                </div>
                <div>
                  <div className="text-sm font-medium text-white">{tx('Manage Schemes', 'योजनाओं का प्रबंधन')}</div>
                  <div className="text-xs text-slate-500">{tx('Add or edit projects', 'परियोजनाएं जोड़ें या संपादित करें')}</div>
                </div>
              </button>

              <button 
                onClick={() => setActiveView('reports')}
                className="flex items-center gap-3 p-3 rounded-lg bg-slate-800/50 hover:bg-blue-600/20 border border-white/5 hover:border-blue-500/30 transition-all group text-left"
              >
                <div className="p-2 rounded-md bg-blue-500/20 text-blue-400 group-hover:bg-blue-500 group-hover:text-white transition-colors">
                  <MessageSquare size={16} />
                </div>
                <div>
                  <div className="text-sm font-medium text-white">{tx('Citizen Feedback', 'नागरिक फीडबैक')}</div>
                  <div className="text-xs text-slate-500">{tx('Review community input', 'समुदाय की प्रतिक्रिया देखें')}</div>
                </div>
              </button>

              <button 
                onClick={() => setShowRagModal(true)}
                className="flex items-center gap-3 p-3 rounded-lg bg-slate-800/50 hover:bg-emerald-600/20 border border-white/5 hover:border-emerald-500/30 transition-all group text-left"
              >
                <div className="p-2 rounded-md bg-emerald-500/20 text-emerald-400 group-hover:bg-emerald-500 group-hover:text-white transition-colors">
                  <Sparkles size={16} />
                </div>
                <div>
                  <div className="text-sm font-medium text-white">{tx('AI Insights', 'एआई इनसाइट्स')}</div>
                  <div className="text-xs text-slate-500">{tx('Ask questions about data', 'डेटा के बारे में प्रश्न पूछें')}</div>
                </div>
              </button>
            </div>
          </div>

          {/* Citizen Sentiment Mini-Card */}
          <div className="bg-gradient-to-br from-indigo-900/40 to-purple-900/40 border border-white/10 rounded-xl p-5">
            <div className="flex items-center justify-between mb-2">
              <h3 className="font-bold text-white text-sm">{tx('Citizen Sentiment', 'नागरिक भावना')}</h3>
              <Users size={16} className="text-indigo-300" />
            </div>
            <div className="flex items-end gap-2 mb-2">
              <span className="text-3xl font-bold text-white">{avgRating}</span>
              <div className="flex mb-1.5">
                {[1, 2, 3, 4, 5].map((star) => (
                  <Star key={star} size={12} className={`${star <= Math.round(parseFloat(avgRating)) ? 'text-yellow-400 fill-yellow-400' : 'text-slate-600'}`} />
                ))}
              </div>
            </div>
            <div className="text-xs text-indigo-200">
              {tx('Based on', 'आधारित')} {totalFeedback} {tx('verified reviews', 'सत्यापित समीक्षाओं पर')}
            </div>
          </div>

        </div>
      </div>

      {/* RAG Query Modal */}
      {showRagModal && (
        <RagQueryModal
          isOpen={showRagModal}
          onClose={() => setShowRagModal(false)}
          onHighlightCitation={(citation: Citation) => {
            console.log('📍 Admin Dashboard - Highlight citation:', citation);
            alert(hi ? `📍 एआई संदर्भ:\n\nप्रकार: ${citation.type}\nअंश: ${citation.snippet}\nस्कोर: ${citation.score}` : `📍 Citation from AI:\n\nType: ${citation.type}\nSnippet: ${citation.snippet}\nScore: ${citation.score}`);
          }}
        />
      )}
    </div>
  );
}
