function export_synthetic_data_to_csv()
    % ---- CONFIG ----
    DATASET_ROOT = "C:\Users\ASUS\Desktop\Projects\ML Project\Dataset";
    OUT_ROOT = "C:\Users\ASUS\Desktop\Projects\ML Project";

    IN_ROOT  = fullfile(DATASET_ROOT, "Matlab_Data", "Synthetic");
    OUT_DIR  = fullfile(OUT_ROOT, "Exports");
    OUT_CSV  = fullfile(OUT_DIR, "lit_synth_1s_states.csv");
    BIN_S    = 1;   % 1-second LF bins

    if ~isfolder(OUT_DIR), mkdir(OUT_DIR); end

    % Find all Waveform*.mat under Synthetic
    files = dir(fullfile(IN_ROOT, "**", "Waveform*.mat"));
    if isempty(files)
        error("No Waveform*.mat files found under: %s", IN_ROOT);
    end
    fprintf("Found %d waveform .mat files\n", length(files));

    % We'll build rows incrementally (simple + robust)
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
        i = S.iHall(:);      % choose iHall; swap to iShunt if needed
        p = v .* i;

        sps = double(S.sps);
        binN = round(sps * BIN_S);
        Kbins = floor(length(p)/binN);
        if Kbins < 2
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
            % no events; still export P with no appliance columns
            labels_unique = strings(0,1);
        else
            lab_ev = lab(ev_idx);
            labels_unique = unique(lab_ev);
        end

        % Build one state column per appliance label in this file
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

        % Build table rows for this file
        t_sec = (0:Kbins-1)' * BIN_S;

        % File ID columns (helpful for grouping/splitting later)
        fileRel = string(erase(fpath, DATASET_ROOT + filesep));
        fileCol = repmat(fileRel, Kbins, 1);

        T = table(fileCol, t_sec, P, 'VariableNames', {'file','t_sec','P'});

        for j = 1:length(labels_unique)
            colName = "y_" + labels_unique(j);
            T.(colName) = stateCols(:,j);
        end

        % Harmonize columns across files (union of all columns)
        if isempty(allColNames)
            allColNames = T.Properties.VariableNames;
            allRows = T;
        else
            % add missing columns to existing table
            missingInAll = setdiff(T.Properties.VariableNames, allColNames);
            for m = 1:length(missingInAll)
                allRows.(missingInAll{m}) = zeros(height(allRows),1);
            end

            % add missing columns to current table
            missingInT = setdiff(allColNames, T.Properties.VariableNames);
            for m = 1:length(missingInT)
                T.(missingInT{m}) = zeros(height(T),1);
            end

            % reorder columns and append
            T = T(:, allRows.Properties.VariableNames);
            allRows = [allRows; T];
            allColNames = allRows.Properties.VariableNames;
        end

        if mod(k,50)==0
            fprintf("Processed %d/%d files...\n", k, length(files));
        end
    end

    % Write CSV
    writetable(allRows, OUT_CSV);
    fprintf("Wrote: %s\nRows=%d, Cols=%d\n", OUT_CSV, height(allRows), width(allRows));
end
