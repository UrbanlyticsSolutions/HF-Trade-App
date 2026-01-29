interface StatsCardProps {
  label: string;
  value: string;
  subtext?: string;
  variant?: 'default' | 'positive' | 'negative';
}

export function StatsCard({ label, value, subtext, variant = 'default' }: StatsCardProps) {
  const getValueColor = () => {
    if (variant === 'positive') return '#22c55e';
    if (variant === 'negative') return '#ef4444';
    return '#e2e8f0';
  };

  return (
    <div 
      className="p-6 rounded-xl"
      style={{ background: '#1e293b', border: '1px solid #334155' }}
    >
      <div 
        className="text-sm font-medium mb-2.5"
        style={{ color: '#94a3b8' }}
      >
        {label}
      </div>
      <div 
        className="text-3xl font-bold"
        style={{ color: getValueColor() }}
      >
        {value}
      </div>
      {subtext && (
        <div 
          className="text-xs mt-2"
          style={{ color: '#64748b' }}
        >
          {subtext}
        </div>
      )}
    </div>
  );
}
