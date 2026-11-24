-- Runs once, on first start of an empty Postgres volume.
CREATE TYPE alert_status AS ENUM ('open', 'confirmed_fraud', 'false_positive');

CREATE TABLE alerts (
    tx_id          UUID          PRIMARY KEY,
    card_id        TEXT          NOT NULL CHECK (card_id ~ '^[0-9a-f]{16}$'),
    merchant       TEXT          NOT NULL CHECK (length(merchant) BETWEEN 1 AND 128),
    category       TEXT          NOT NULL CHECK (category ~ '^[a-z][a-z_]{0,63}$'),
    score          REAL          NOT NULL CHECK (score >= 0 AND score <= 1),
    amount         NUMERIC(12,2) NOT NULL CHECK (amount >= 0),
    model_version  TEXT          NOT NULL CHECK (length(model_version) BETWEEN 1 AND 128),
    event_time     TIMESTAMPTZ   NOT NULL,
    flagged_at     TIMESTAMPTZ   NOT NULL DEFAULT now(),
    status         alert_status  NOT NULL DEFAULT 'open',
    true_label     BOOLEAN,
    reviewed_at    TIMESTAMPTZ,
    CONSTRAINT reviewed_has_time CHECK ((status = 'open') = (reviewed_at IS NULL))
);

CREATE INDEX alerts_status_flagged_idx ON alerts (status, flagged_at DESC);
CREATE INDEX alerts_flagged_idx ON alerts (flagged_at DESC, tx_id DESC);
CREATE INDEX alerts_card_idx ON alerts (card_id, flagged_at DESC);
