import "./Primitives.css";

export function PageHeader({ eyebrow, title, description, actions }) {
  return (
    <header className="page-header">
      <div>
        {eyebrow && <div className="page-eyebrow">{eyebrow}</div>}
        <h1 className="page-title">{title}</h1>
        {description && <p className="page-desc">{description}</p>}
      </div>
      {actions && <div className="page-actions">{actions}</div>}
    </header>
  );
}

export function Panel({ title, subtitle, children, className = "", right }) {
  return (
    <section className={`panel ${className}`}>
      {(title || right) && (
        <div className="panel-head">
          <div>
            {title && <h2 className="panel-title">{title}</h2>}
            {subtitle && <p className="panel-subtitle">{subtitle}</p>}
          </div>
          {right}
        </div>
      )}
      <div className="panel-body">{children}</div>
    </section>
  );
}

export function Stat({ label, value, unit, tone = "neutral" }) {
  return (
    <div className="stat">
      <span className="stat-label">{label}</span>
      <span className={`stat-value tone-${tone}`}>
        {value}
        {unit && <span className="stat-unit">{unit}</span>}
      </span>
    </div>
  );
}

export function Badge({ children, tone = "neutral" }) {
  return <span className={`badge tone-${tone}`}>{children}</span>;
}

export function ParadigmTag({ paradigm }) {
  const isDl = paradigm === "deep_learning";
  return (
    <span className={`paradigm-tag ${isDl ? "paradigm-dl" : "paradigm-trad"}`}>
      {isDl ? "DL" : "Trad"}
    </span>
  );
}

export function EmptyState({ title, description, action }) {
  return (
    <div className="empty-state">
      <div className="empty-state-mark">—</div>
      <h3>{title}</h3>
      {description && <p>{description}</p>}
      {action}
    </div>
  );
}

export function Spinner() {
  return <div className="spinner" aria-label="Loading" />;
}
