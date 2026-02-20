function export_simulated_data_to_csv()
    % ---- CONFIG ----
    DATASET_ROOT = "C:\Users\ASUS\Desktop\Projects\ML Project\Dataset";
    OUT_ROOT = "C:\Users\ASUS\Desktop\Projects\ML Project";
    % Simulated waveforms live here after bin->mat conversion
    IN_ROOT  = fullfile(DATASET_ROOT, "Matlab_Data", "Simulated");
    
    % We will build label->appliance-name mapping from Synthetic metadata
    SYN_ROOT = fullfile(DATASET_ROOT, "Matlab_Data", "Synthetic");
    
    OUT_DIR  = fullfile(OUT_ROOT, "Exports");
    OUT_CSV  = fullfile(OUT_DIR, "simulated_data_1s_states.csv");
    
    BIN_S    = 1;   % 1-second LF bins

    if ~isfolder(OUT_DIR), mkdir(OUT_DIR); end
    if ~isfolder(IN_ROOT)
        error("Simulated Matlab_Data folder not found: %s", IN_ROOT);
    end
    if ~isfolder(SYN_ROOT)
        error("Synthetic Matlab_Data folder not found: %s", SYN_ROOT);
    end

    % ------------------------------------------------------------
    % 1) Build mapping from label (A0, B0, ...) -> real appliance name
    % ------------------------------------------------------------
    label2name = build_label_name_map_from_synthetic(SYN_ROOT);

    % ------------------------------------------------------------
    % 2) Export Simulated Waveform*.mat
    % ------------------------------------------------------------
    files = dir(fullfile(IN_ROOT, "**", "Waveform*.mat"));
    if isempty(files)
        error("No Waveform*.mat files found under: %s", IN_ROOT);
    end
    fprintf("Found %d simulated waveform .mat files\n", length(files));

    allRows = [];
    allColNames = {};

    for k = 1:length(files)
        fpath = fullfile(files(k).folder, files(k).name);
        S = load(fpath);

        % Required vars
        if ~isfield(S,'vGrid') || ~isfield(S,'iHall') || ~isfield(S,'events_r') || ~isfield(S,'labels') || ~isfield(S,'sps')
            fprintf("Skipping (missing vars): %s\n", fpath);
            continue;
        end

        v = S.vGrid(:);
        i = S.iHall(:);      % switch to iShunt if you want
        p = v .* i;

        sps = double(S.sps);
        binN = round(sps * BIN_S);
        Kbins = floor(length(p)/binN);
        if Kbins < 1
            fprintf("Skipping (too short): %s\n", fpath);
            continue;
        end

        % LF aggregate power
        P = mean(reshape(p(1:Kbins*binN), binN, Kbins), 1)';  % Kbins x 1

        % Sample-level appliance states from events+labels
        ev = S.events_r(:);
        lab = S.labels;

        ev_idx = find(ev ~= 0);
        if isempty(ev_idx)
            labels_unique = strings(0,1);
        else
            lab_ev = lab(ev_idx);
            labels_unique = unique(lab_ev);
        end

        stateCols = zeros(Kbins, length(labels_unique));
        for j = 1:length(labels_unique)
            lj = labels_unique(j);
            e = zeros(length(ev),1);
            e(ev_idx(lab_ev==lj)) = ev(ev_idx(lab_ev==lj));
            st = cumsum(e);
            st = max(min(st,1),0);

            st_lf = mean(reshape(st(1:Kbins*binN), binN, Kbins), 1)' > 0.5;
            stateCols(:,j) = double(st_lf);
        end

        % Time
        t_sec = (0:Kbins-1)' * BIN_S;

        % File ID (keep subset path)
        fileRel = string(erase(fpath, DATASET_ROOT + filesep));
        fileCol = repmat(fileRel, Kbins, 1);

        T = table(fileCol, t_sec, P, 'VariableNames', {'file','t_sec','P'});

        % Add label columns with REAL appliance names (duplicate-safe)
        for j = 1:length(labels_unique)
            lbl = string(labels_unique(j));  % e.g., "A0"
            if isKey(label2name, char(lbl))
                nice = string(label2name(char(lbl)));
            else
                nice = "Unknown_" + lbl; % fallback
            end

            safeNice = make_safe_colname(nice);
            colName = "y_" + safeNice + "__" + lbl;  % keep original label to avoid collisions
            T.(colName) = stateCols(:,j);
        end

        % Harmonize columns across files (union)
        if isempty(allColNames)
            allColNames = T.Properties.VariableNames;
            allRows = T;
        else
            missingInAll = setdiff(T.Properties.VariableNames, allColNames);
            for m = 1:length(missingInAll)
                allRows.(missingInAll{m}) = zeros(height(allRows),1);
            end

            missingInT = setdiff(allColNames, T.Properties.VariableNames);
            for m = 1:length(missingInT)
                T.(missingInT{m}) = zeros(height(T),1);
            end

            T = T(:, allRows.Properties.VariableNames);
            allRows = [allRows; T];
            allColNames = allRows.Properties.VariableNames;
        end

        if mod(k,50)==0
            fprintf("Processed %d/%d files...\n", k, length(files));
        end
    end

    writetable(allRows, OUT_CSV);
    fprintf("Wrote: %s\nRows=%d, Cols=%d\n", OUT_CSV, height(allRows), width(allRows));
end

% ===================== Helpers =====================

function label2name = build_label_name_map_from_synthetic(SYN_ROOT)
    % Build map like: "A0" -> "Microwave, Consul ... standby 4.5W"
    files = dir(fullfile(SYN_ROOT, "**", "Waveform*.mat"));
    if isempty(files)
        error("No Waveform*.mat files found under Synthetic: %s", SYN_ROOT);
    end

    label2name = containers.Map('KeyType','char','ValueType','char');

    for k = 1:length(files)
        fpath = fullfile(files(k).folder, files(k).name);
        S = load(fpath);

        if ~isfield(S,'load_descr') || ~isfield(S,'load_descr_short')
            continue;
        end

        % In synthetic, load_descr_short looks like "1A0", "2Q0R0", etc.
        short = string(S.load_descr_short);
        % Extract all label tokens like A0, Q0, R0, ...
        toks = regexp(short, "[A-Z]0", "match");
        if isempty(toks)
            continue;
        end

        desc = string(S.load_descr);

        for t = 1:length(toks)
            key = char(toks{t});
            if ~isKey(label2name, key)
                label2name(key) = char(desc);
            end
        end

        % Stop early once we have all 26 labels
        if label2name.Count >= 26
            break;
        end
    end

    fprintf("Found mappings: %d\n", label2name.Count);
end

function out = make_safe_colname(s)
    % Turn description into CSV-friendly column token
    s = string(s);
    s = replace(s, newline, " ");
    s = regexprep(s, "\s+", " ");
    s = strtrim(s);

    % Replace non-alphanum with underscores
    s = regexprep(s, "[^A-Za-z0-9]+", "_");
    s = regexprep(s, "_+", "_");
    s = regexprep(s, "^_|_$", "");

    % Avoid empty
    if strlength(s) == 0
        s = "Unknown";
    end

    out = s;
end
