export function LoadingOverlay({ message }: { message: string }) {
  return (
    <div 
      className="fixed inset-0 flex items-center justify-center z-50"
      style={{ background: 'rgba(15, 23, 42, 0.9)' }}
    >
      <div 
        className="p-8 rounded-xl text-center"
        style={{ background: '#1e293b', border: '1px solid #334155' }}
      >
        <div className="flex justify-center mb-4">
          <div 
            className="w-12 h-12 border-4 border-t-transparent rounded-full animate-spin"
            style={{ borderColor: '#38bdf8', borderTopColor: 'transparent' }}
          />
        </div>
        <p 
          className="text-base font-medium"
          style={{ color: '#e2e8f0' }}
        >
          {message}
        </p>
      </div>
    </div>
  );
}
