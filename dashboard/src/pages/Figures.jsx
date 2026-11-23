import { useState } from "react";
import { PageHeader, Panel, Spinner, EmptyState, ErrorState, Notice } from "../components/PageHeader";
import api from "../api/client";
import useApi from "../api/useApi";

export default function Figures({ runId }) {
  const [lightbox, setLightbox] = useState(null);

  // Prefer figures stored under this run's id; otherwise show the whole figures
  // directory and say so, since those figures are not tied to the selected run.
  const res = useApi(async () => {
    if (runId) {
      const scoped = await api.listFigures(runId);
      if (scoped.figures.length) return { figures: scoped.figures, scoped: true };
    }
    const all = await api.listFigures("");
    return { figures: all.figures, scoped: false };
  }, [runId]);

  const figures = res.data?.figures || [];
  const scoped = !!res.data?.scoped;

  const grouped = figures.reduce((acc, f) => {
    (acc[f.figure_type] ||= []).push(f);
    return acc;
  }, {});

  return (
    <>
      <PageHeader
        eyebrow="Publication-quality exports"
        title="Figures"
        description="PNG/SVG figures found in the figures directory. Click to enlarge."
      />

      {!res.loading && !res.error && figures.length > 0 && !scoped && (
        <Notice>
          No figures are stored under this run's id, so every figure in the figures directory is
          shown. They are not necessarily from the selected run.
        </Notice>
      )}

      {res.loading ? (
        <Panel>
          <Spinner />
        </Panel>
      ) : res.error ? (
        <Panel><ErrorState title="Could not load figures" error={res.error} /></Panel>
      ) : figures.length === 0 ? (
        <Panel>
          <EmptyState
            title="No figures generated yet"
            description="No PNG files were found in the figures directory (FACEEVAL_FIGURES_DIR, default reports/figures)."
          />
        </Panel>
      ) : (
        Object.entries(grouped).map(([type, figs]) => (
          <Panel key={type} title={type.replace(/_/g, " ")} subtitle={`${figs.length} figure(s)`}>
            <div className="figure-grid">
              {figs.map((f) => (
                <button
                  key={f.png_url}
                  className="figure-card"
                  onClick={() => setLightbox(f)}
                >
                  <img src={api.assetUrl(f.png_url)} alt={f.title} loading="lazy" />
                  <span className="figure-card-title">{f.title}</span>
                </button>
              ))}
            </div>
          </Panel>
        ))
      )}

      {lightbox && (
        <div className="lightbox" onClick={() => setLightbox(null)}>
          <div className="lightbox-inner" onClick={(e) => e.stopPropagation()}>
            <img src={api.assetUrl(lightbox.png_url)} alt={lightbox.title} />
            <div className="lightbox-foot">
              <span>{lightbox.title}</span>
              <div className="lightbox-actions">
                {lightbox.svg_url && (
                  <a href={api.assetUrl(lightbox.svg_url)} download className="lightbox-link">
                    Download SVG
                  </a>
                )}
                <a href={api.assetUrl(lightbox.png_url)} download className="lightbox-link">
                  Download PNG
                </a>
                <button className="lightbox-close" onClick={() => setLightbox(null)}>
                  Close
                </button>
              </div>
            </div>
          </div>
        </div>
      )}
    </>
  );
}
