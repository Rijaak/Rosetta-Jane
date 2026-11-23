package org.rosettajane.app;

import android.Manifest;
import android.app.Activity;
import android.content.ContentResolver;
import android.content.Intent;
import android.content.SharedPreferences;
import android.content.pm.ApplicationInfo;
import android.database.Cursor;
import android.net.Uri;
import android.os.Build;
import android.os.Bundle;
import android.os.Handler;
import android.os.Looper;
import android.provider.DocumentsContract;
import android.text.InputType;
import android.util.Log;
import android.view.View;
import android.view.ViewGroup;
import android.widget.ArrayAdapter;
import android.widget.Button;
import android.widget.CheckBox;
import android.widget.EditText;
import android.widget.LinearLayout;
import android.widget.ScrollView;
import android.widget.Spinner;
import android.widget.TextView;

import org.json.JSONArray;
import org.json.JSONObject;

import java.io.File;
import java.io.FileOutputStream;
import java.io.InputStream;
import java.io.OutputStream;
import java.util.ArrayList;
import java.util.List;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;

/** One screen: read the warning, pick a model and a group, switch the p2p service on or off. */
public class MainActivity extends Activity {
    private static final String TAG = "RosettaJane";
    private static final int PICK_FOLDER = 7;
    private static final int PICK_PAIRS = 8;
    private static final int SAVE_ADAPTER = 9;
    private static final int PICK_SHARED = 10;
    private static final int SAVE_SCORES = 11;
    private static final String WARNING =
            "Before you run the p2p service\n\n"
            + "• It is safest inside a group code, where every peer is someone you chose. On the "
            + "\"public\" network, use is open to anyone.\n"
            + "• Open P2P runs strangers' prompts. This device computes layers for whatever anyone "
            + "on the network asks, and cannot see or filter what that is.\n"
            + "• This device never sees the text. It receives processed numbers from partway through "
            + "the model, not words, and does not save them. That is not a strong privacy guarantee for "
            + "the person asking, and it does not tell you what this device helped answer.\n"
            + "• This device is reachable by other devices. The service opens a port (7700). On mobile "
            + "data most carriers block incoming connections, so it usually only works on Wi-Fi or through "
            + "a mesh VPN such as Tailscale. A bug in this program could expose the device.\n"
            + "• It uses this device's processor, battery and data whenever it is on. Expect heat and "
            + "battery drain.\n"
            + "• Check the terms of the model you load: the service runs it for other people.\n"
            + "• No warranty. This software is provided as is, under the GNU Affero General Public License, version 3.";

    private final Handler ui = new Handler(Looper.getMainLooper());
    private final ExecutorService work = Executors.newSingleThreadExecutor();
    private final List<String> modelPaths = new ArrayList<>();
    private Spinner modelPicker;
    private EditText groupCode, peers, layers, span;
    private CheckBox training, understood;
    private Button toggle, importButton;
    private TextView status, modelNote;
    private boolean running;
    private EditText ownLayers, question, runName, passes;
    private Button askButton, trainButton, pairsButton;
    private TextView answer, trainStatus, pairsNote;
    private LinearLayout adapterList;
    // Teams: an adapter used together because people signed on. Drawn from what the Python side says.
    private TextView teamNote;
    private EditText teamCode, teamName;
    private LinearLayout teamButtons;
    private String shownTeamButtons = "", trainFrom = "";
    private boolean asking, training_on;
    private String shownAdapters = "", savingAdapter, lastSide = "";
    private File pairsFile;
    // Trying an adapter: every question answered twice, without it and with it. Nothing is applied.
    private String trialOf;
    private boolean trialChat, trialBusy;
    private final List<String[]> trialTurns = new ArrayList<>();      // question, without, with
    private final List<String> trialPending = new ArrayList<>();
    private TextView trialTitle, trialOut, sideNote;
    private EditText trialText, trialLength;
    private Button trialButton, trialModeButton;
    private LinearLayout trialPanel;
    private SharedPreferences prefs;

    @Override
    protected void onCreate(Bundle saved) {
        super.onCreate(saved);
        prefs = getSharedPreferences("settings", MODE_PRIVATE);
        int pad = (int) (16 * getResources().getDisplayMetrics().density);
        LinearLayout column = new LinearLayout(this);
        column.setOrientation(LinearLayout.VERTICAL);
        column.setPadding(pad, pad, pad, pad);
        ScrollView scroll = new ScrollView(this);
        scroll.addView(column);
        setContentView(scroll);

        TextView warning = text(WARNING, 14);
        warning.setBackgroundColor(0xFFFFF4D6);
        warning.setPadding(pad, pad, pad, pad);
        column.addView(warning);

        column.addView(label("Model"));
        modelPicker = new Spinner(this);
        column.addView(modelPicker);
        modelNote = text("", 13);
        column.addView(modelNote);
        importButton = new Button(this);
        importButton.setText("Import a model folder");
        importButton.setOnClickListener(v -> startActivityForResult(
                new Intent(Intent.ACTION_OPEN_DOCUMENT_TREE), PICK_FOLDER));
        column.addView(importButton);

        column.addView(label("Group code (everyone in your group enters the same one)"));
        groupCode = field(prefs.getString("group", ""), InputType.TYPE_CLASS_TEXT
                | InputType.TYPE_TEXT_VARIATION_VISIBLE_PASSWORD);
        column.addView(groupCode);
        column.addView(label("Address of any device already in the group (IP:port)"));
        peers = field(prefs.getString("peers", ""), InputType.TYPE_CLASS_TEXT | InputType.TYPE_TEXT_VARIATION_URI);
        column.addView(peers);
        column.addView(label("Layers to serve: \"auto\" takes what the group is shortest of, or FIRST:END"));
        layers = field(prefs.getString("layers", "auto"), InputType.TYPE_CLASS_TEXT);
        column.addView(layers);
        column.addView(label("With auto: how many layers this device can hold"));
        span = field(prefs.getString("span", "2"), InputType.TYPE_CLASS_NUMBER);
        column.addView(span);

        training = new CheckBox(this);
        training.setText("Also help others train (heavier work)");
        training.setChecked(prefs.getBoolean("training", false));
        column.addView(training);
        understood = new CheckBox(this);
        understood.setText("I have read the warning above");
        column.addView(understood);

        toggle = new Button(this);
        toggle.setOnClickListener(v -> {
            if (running) {
                stopService();
            } else {
                startFromScreen();
            }
        });
        column.addView(toggle);
        status = text("", 15);
        status.setPadding(0, pad, 0, pad);
        status.setTextIsSelectable(true);
        column.addView(status);

        column.addView(heading("Ask"));
        column.addView(text("Your question is answered through the group. This device keeps the first layers and "
                + "turns the reply back into words; the devices that help receive an encoded form of your whole "
                + "question, which can be turned back into an approximation of it.", 13));
        column.addView(label("Layers kept on this device (more is more private, and slower here)"));
        ownLayers = field(prefs.getString("own", "2"), InputType.TYPE_CLASS_NUMBER);
        column.addView(ownLayers);
        column.addView(label("Question"));
        question = field("", InputType.TYPE_CLASS_TEXT | InputType.TYPE_TEXT_FLAG_MULTI_LINE);
        question.setMinLines(2);
        column.addView(question);
        askButton = new Button(this);
        askButton.setOnClickListener(v -> {
            if (asking) {
                work.execute(() -> PeerService.bridge(this).callAttr("ask_stop"));
            } else {
                ask(question.getText().toString().trim());
            }
        });
        column.addView(askButton);
        answer = text("", 16);
        answer.setTextIsSelectable(true);
        answer.setPadding(0, pad / 2, 0, pad);
        column.addView(answer);

        column.addView(heading("Train"));
        column.addView(text("Teach the model from your own question and answer pairs using the group's hardware. "
                + "This device keeps every trained weight; other devices only calculate. They can see an encoded "
                + "form of your pairs, so do not train on private material with people you do not trust. "
                + "Training needs devices in your group that have chosen to help others train.", 13));
        pairsButton = new Button(this);
        pairsButton.setText("Choose a file of pairs");
        pairsButton.setOnClickListener(v -> startActivityForResult(new Intent(Intent.ACTION_OPEN_DOCUMENT)
                .addCategory(Intent.CATEGORY_OPENABLE).setType("*/*"), PICK_PAIRS));
        column.addView(pairsButton);
        pairsNote = text("One pair per line: {\"question\": \"...\", \"answer\": \"...\"}", 13);
        column.addView(pairsNote);
        column.addView(label("Name for this run (lowercase letters, digits, dashes)"));
        runName = field(prefs.getString("run", "my-first-run"), InputType.TYPE_CLASS_TEXT);
        column.addView(runName);
        column.addView(label("Passes over your pairs"));
        passes = field("3", InputType.TYPE_CLASS_NUMBER);
        column.addView(passes);
        trainButton = new Button(this);
        trainButton.setOnClickListener(v -> {
            if (training_on) {
                work.execute(() -> PeerService.bridge(this).callAttr("train_stop"));
            } else {
                train(pairsFile, runName.getText().toString().trim(), passes.getText().toString().trim());
            }
        });
        column.addView(trainButton);
        trainStatus = text("", 15);
        trainStatus.setPadding(0, pad / 2, 0, pad);
        column.addView(trainStatus);

        column.addView(heading("Trained adapters"));
        TextView bump = text("Finished training is never applied for you. A trained adapter changes how the model "
                + "answers, and part of the work was done on other people's devices. Save a copy, read the note "
                + "inside, try it, score it on questions it never saw, and apply it yourself only if "
                + "you are satisfied. Each adapter is its own decision.\n\n"
                + "If you build on Rosetta-Jane, keep this step. Removing it trades a minute of someone's attention "
                + "for the one safeguard that does not depend on the network behaving.", 13);
        bump.setBackgroundColor(0xFFE8EEF7);
        bump.setPadding(pad, pad, pad, pad);
        column.addView(bump);
        adapterList = new LinearLayout(this);
        adapterList.setOrientation(LinearLayout.VERTICAL);
        column.addView(adapterList);
        sideNote = text("", 14);
        sideNote.setTextIsSelectable(true);
        column.addView(sideNote);
        Button scores = new Button(this);
        scores.setText("Save Model Adapter Score\u2026");
        scores.setOnClickListener(v -> startActivityForResult(new Intent(Intent.ACTION_CREATE_DOCUMENT)
                .addCategory(Intent.CATEGORY_OPENABLE).setType("text/html")
                .putExtra(Intent.EXTRA_TITLE, "Model Adapter Score.html"), SAVE_SCORES));
        column.addView(scores);
        column.addView(text("One landscape page with every score on this phone: each adapter beside the unchanged "
                + "model, grouped by model, best first. It opens in any browser and is sent nowhere.", 13));
        Button addShared = new Button(this);
        addShared.setText("Add an adapter someone shared");
        addShared.setOnClickListener(v -> startActivityForResult(new Intent(Intent.ACTION_OPEN_DOCUMENT)
                .addCategory(Intent.CATEGORY_OPENABLE).setType("application/zip"), PICK_SHARED));
        column.addView(addShared);
        column.addView(text("Pick the zip someone sent you. It is checked and stored so you can try it and score "
                + "it on your own questions. Adding it applies nothing.", 13));

        trialPanel = new LinearLayout(this);
        trialPanel.setOrientation(LinearLayout.VERTICAL);
        trialPanel.setVisibility(View.GONE);
        trialTitle = heading("");
        trialPanel.addView(trialTitle);
        trialPanel.addView(text("Every question is answered twice: by the unchanged model, and by the model with "
                + "this adapter. This is a trial. The adapter is not applied to anything. Trials are slow, and need "
                + "devices that have chosen to help others train.", 13));
        trialModeButton = new Button(this);
        trialModeButton.setOnClickListener(v -> {
            trialChat = !trialChat;
            trialTurns.clear();
            trialPending.clear();
            showTrialMode();
        });
        trialPanel.addView(trialModeButton);
        trialPanel.addView(label("Reply length in tokens (48 is about 35 words; longer is slower)"));
        trialLength = field("48", InputType.TYPE_CLASS_NUMBER);
        trialPanel.addView(trialLength);
        trialText = field("", InputType.TYPE_CLASS_TEXT | InputType.TYPE_TEXT_FLAG_MULTI_LINE);
        trialText.setMinLines(3);
        trialPanel.addView(trialText);
        trialButton = new Button(this);
        trialButton.setOnClickListener(v -> {
            if (trialBusy) {
                work.execute(() -> PeerService.bridge(this).callAttr("adapter_stop"));
            } else {
                startTrial();
            }
        });
        trialPanel.addView(trialButton);
        trialOut = text("", 15);
        trialOut.setTextIsSelectable(true);
        trialOut.setPadding(0, pad / 2, 0, pad);
        trialPanel.addView(trialOut);
        column.addView(trialPanel);
        showTrialMode();

        column.addView(heading("Teams: use an adapter together"));
        TextView teams = text("A team is how a group chooses to pool its devices, the way players join a co-op game. "
                + "Anyone can lead one: with an adapter they hold (press \"Lead a team with it\" on an adapter above), "
                + "or with no adapter, to share computing power for the unchanged model. The leader chooses who can "
                + "sign on: anyone in the group, people who ask and are approved, or only people with the team's code "
                + "or an invitation. You are shown the model and the adapter before you join, because everyone in a "
                + "team uses the same ones. Say yes to an invitation, or have your request approved, and the adapter "
                + "is supplied at once. Once the leader begins, the team's answers come from the team. Leave, or have "
                + "the leader end the team, and it is the unchanged model again.\n\n"
                + "Nothing is applied on a device whose owner did not sign it on. If you build on Rosetta-Jane, keep "
                + "the signing on.\n\nThe team screens are modelled on the co-op in Yharnam Together. Thanks to Kaniel, "
                + "AMANISES (CLS_AMG), Supermedo and everyone who built it.", 13);
        teams.setBackgroundColor(0xFFE8EEF7);
        teams.setPadding(pad, pad, pad, pad);
        column.addView(teams);
        column.addView(label("Your name on team screens (optional). It is shown with a short tag made from this "
                + "phone's own key, so people you team with can tell it is this phone again."));
        teamName = field(prefs.getString("called", ""), InputType.TYPE_CLASS_TEXT);
        column.addView(teamName);
        column.addView(label("Team code (six letters and digits, like DAF H2D)"));
        teamCode = field("", InputType.TYPE_CLASS_TEXT | InputType.TYPE_TEXT_FLAG_CAP_CHARACTERS);
        column.addView(teamCode);
        teamNote = text("", 15);
        teamNote.setTextIsSelectable(true);
        teamNote.setPadding(0, pad / 2, 0, pad / 2);
        column.addView(teamNote);
        teamButtons = new LinearLayout(this);
        teamButtons.setOrientation(LinearLayout.VERTICAL);
        column.addView(teamButtons);

        if (Build.VERSION.SDK_INT >= 33) {
            requestPermissions(new String[]{Manifest.permission.POST_NOTIFICATIONS}, 1);
        }
        refreshModels();
        automate(getIntent());
        tick();
    }

    private TextView text(String s, int size) {
        TextView t = new TextView(this);
        t.setText(s);
        t.setTextSize(size);
        return t;
    }

    private TextView heading(String s) {
        TextView t = text(s, 20);
        t.setTextColor(0xFF424C36);
        t.setPadding(0, (int) (28 * getResources().getDisplayMetrics().density), 0,
                (int) (6 * getResources().getDisplayMetrics().density));
        return t;
    }

    /** The model, group code and peer address from the top of the screen, or null with a message shown. */
    private String[] connection(TextView where) {
        String code = groupCode.getText().toString().trim();
        int chosen = modelPicker.getSelectedItemPosition();
        if (chosen < 0 || chosen >= modelPaths.size()) {
            where.setText("Import a model first.");
            return null;
        }
        if (code.replace(" ", "").length() < 8) {
            where.setText("Enter your group code at the top first (at least 8 characters).");
            return null;
        }
        String own = ownLayers.getText().toString().trim();
        prefs.edit().putString("group", code).putString("peers", peers.getText().toString().trim())
                .putString("own", own).putString("model", modelPaths.get(chosen)).apply();
        return new String[]{modelPaths.get(chosen), code, peers.getText().toString().trim(), own.isEmpty() ? "2" : own};
    }

    private void ask(String text) {
        String[] c = connection(answer);
        if (c == null) {
            return;
        }
        if (text.isEmpty()) {
            answer.setText("Type a question first.");
            return;
        }
        askWith(c[0], c[1], c[2], Integer.parseInt(c[3]), text, 256, 0.7);
    }

    private void askWith(String model, String code, String peerList, int own, String text, int limit, double heat) {
        work.execute(() -> {
            try {
                JSONArray messages = new JSONArray().put(new JSONObject().put("role", "user").put("content", text));
                PeerService.bridge(this).callAttr("ask_start", model, code, peerList.split("[,\\s]+"),
                        messages.toString(), limit, heat, own);
            } catch (Exception e) {
                Log.e(TAG, "ask", e);
            }
        });
    }

    private void train(File pairs, String name, String rounds) {
        String[] c = connection(trainStatus);
        if (c == null) {
            return;
        }
        if (pairs == null || !pairs.exists()) {
            trainStatus.setText("Choose a file of pairs first.");
            return;
        }
        double epochs;
        try {
            epochs = Math.max(0.1, Double.parseDouble(rounds));
        } catch (NumberFormatException e) {
            epochs = 3;
        }
        prefs.edit().putString("run", name).apply();
        trainWith(c[0], c[1], c[2], Integer.parseInt(c[3]), pairs.getAbsolutePath(), name, epochs);
    }

    private void trainWith(String model, String code, String peerList, int own, String pairs, String name, double epochs) {
        final String from = trainFrom;
        trainFrom = "";
        work.execute(() -> PeerService.bridge(this).callAttr("train_start", model, code, peerList.split("[,\\s]+"),
                pairs, name, epochs, own, getFilesDir().getAbsolutePath(), from));
    }

    private void showTrialMode() {
        trialModeButton.setText(trialChat ? "Mode: a trial chat (tap for a list of questions)"
                : "Mode: a list of questions (tap for a trial chat)");
        trialText.setHint(trialChat ? "Your next message" : "Questions, one per line");
        trialButton.setText(trialChat ? "Send" : "Try these questions");
        trialOut.setText("");
    }

    private JSONArray side(int which, String question) throws Exception {
        JSONArray messages = new JSONArray();
        if (trialChat) {                            // each side carries its own earlier replies forward
            for (String[] turn : trialTurns) {
                messages.put(new JSONObject().put("role", "user").put("content", turn[0]));
                messages.put(new JSONObject().put("role", "assistant")
                        .put("content", turn[which].isEmpty() ? "(no reply)" : turn[which]));
            }
        }
        return messages.put(new JSONObject().put("role", "user").put("content", question));
    }

    private void startTrial() {
        String[] c = connection(trialOut);
        if (c == null || trialOf == null) {
            return;
        }
        String typed = trialText.getText().toString().trim();
        if (typed.isEmpty()) {
            trialOut.setText("Type a question first.");
            return;
        }
        int limit;
        try {
            limit = Math.max(1, Math.min(512, Integer.parseInt(trialLength.getText().toString().trim())));
        } catch (NumberFormatException e) {
            limit = 48;
        }
        try {
            if (!trialChat) {
                trialTurns.clear();
            }
            trialPending.clear();
            JSONArray items = new JSONArray();
            for (String q : trialChat ? new String[]{typed} : typed.split("\\n")) {
                if (!q.trim().isEmpty()) {
                    trialPending.add(q.trim());
                    items.put(new JSONObject().put("without", side(1, q.trim())).put("with", side(2, q.trim())));
                }
            }
            final String payload = items.toString(), name = trialOf;
            final int cap = limit;
            trialText.setText("");
            work.execute(() -> {
                String reply = PeerService.bridge(this).callAttr("adapter_try", c[0], c[1], c[2].split("[,\\s]+"),
                        Integer.parseInt(c[3]), getFilesDir().getAbsolutePath(), name, payload, cap).toString();
                if (!reply.equals("started")) {
                    ui.post(() -> trialOut.setText(reply.equals("busy")
                            ? "Something else is running. Wait for it to finish, or stop it." : reply));
                }
            });
        } catch (Exception e) {
            trialOut.setText("Could not start the trial: " + e.getMessage());
        }
    }

    private void scoreOnMyPairs(String name) {
        String[] c = connection(sideNote);
        if (c == null) {
            return;
        }
        if (pairsFile == null || !pairsFile.exists()) {
            sideNote.setText("Choose a file of pairs first (under Train). Use questions this adapter was not trained on.");
            return;
        }
        work.execute(() -> {
            String reply = PeerService.bridge(this).callAttr("adapter_score", c[0], c[1], c[2].split("[,\\s]+"),
                    Integer.parseInt(c[3]), getFilesDir().getAbsolutePath(), name, pairsFile.getAbsolutePath()).toString();
            ui.post(() -> sideNote.setText(reply.equals("started") ? "Scoring \"" + name + "\" on your pairs\u2026"
                    : reply.equals("busy") ? "Something else is running. Wait for it to finish, or stop it." : reply));
        });
    }

    private void shareScoreCard(String name) {
        work.execute(() -> {
            try {
                JSONObject card = new JSONObject(PeerService.bridge(this).callAttr("score_card",
                        getFilesDir().getAbsolutePath(), name).toString());
                String shareText = card.optString("share_text");
                StringBuilder lines = new StringBuilder();
                JSONArray shown = card.getJSONArray("lines");
                for (int i = 0; i < shown.length(); i++) {
                    lines.append(shown.getString(i)).append("\n");
                }
                ui.post(() -> {
                    sideNote.setText(lines.toString().trim());
                    if (!shareText.isEmpty()) {     // the person picks where it goes; nothing is sent otherwise
                        startActivity(Intent.createChooser(new Intent(Intent.ACTION_SEND).setType("text/plain")
                                .putExtra(Intent.EXTRA_TEXT, shareText), "Share score card"));
                    }
                });
            } catch (Exception e) {
                Log.e(TAG, "score card", e);
            }
        });
    }

    private Button small(String label, View.OnClickListener action) {
        Button b = new Button(this);
        b.setText(label);
        b.setOnClickListener(action);
        return b;
    }

    /** One press on the Teams part of the screen. The Python side does it; the next refresh shows the result. */
    private void teamDo(String kind, String what, JSONObject req, boolean needsCode) {
        String[] c = connection(teamNote);
        if (c == null) {
            return;
        }
        try {
            if (needsCode) {
                String typed = teamCode.getText().toString().trim();
                if (typed.isEmpty()) {
                    teamNote.setText("Type the team's code first.");
                    return;
                }
                req = new JSONObject(req.toString()).put("code", typed);
            }
            String called = teamName.getText().toString().trim();
            prefs.edit().putString("called", called).apply();
            req = new JSONObject(req.toString()).put("_called", called)
                    .put("_question", question.getText().toString().trim());
        } catch (Exception e) {
            Log.e(TAG, "team code", e);
            return;
        }
        final String body = req.toString();
        teamNote.setText("Working\u2026");
        work.execute(() -> {
            try {
                Log.i(TAG, "TEAM " + kind + " " + what + " [" + PeerService.bridge(this).callAttr("panel_do", c[0], c[1],
                        c[2].split("[,\\s]+"), Integer.parseInt(c[3]), getFilesDir().getAbsolutePath(), kind, what,
                        body) + "]");
            } catch (Exception e) {
                Log.e(TAG, "team", e);
            }
        });
    }

    private void showTeam(JSONObject panel) throws Exception {
        teamNote.setText(panel.getString("text"));
        JSONArray buttons = panel.getJSONArray("buttons");
        if (buttons.toString().equals(shownTeamButtons)) {
            return;
        }
        shownTeamButtons = buttons.toString();
        teamButtons.removeAllViews();
        for (int i = 0; i < buttons.length(); i++) {
            final JSONObject b = buttons.getJSONObject(i);
            teamButtons.addView(small(b.getString("label"), v -> teamDo(b.optString("kind"), b.optString("what"),
                    b.optJSONObject("req") == null ? new JSONObject() : b.optJSONObject("req"),
                    b.optBoolean("needs_code"))));
        }
    }

    private JSONObject named(String name) {
        try {
            return new JSONObject().put("name", name);
        } catch (Exception e) {
            return new JSONObject();
        }
    }

    private void showAdapters(JSONArray found) throws Exception {
        if (found.toString().equals(shownAdapters)) {
            return;
        }
        shownAdapters = found.toString();
        adapterList.removeAllViews();
        if (found.length() == 0) {
            adapterList.addView(text("No adapters yet.", 14));
        }
        for (int i = 0; i < found.length(); i++) {
            JSONObject a = found.getJSONObject(i);
            final String name = a.getString("name");
            TextView title = text(name, 17);
            title.setPadding(0, (int) (14 * getResources().getDisplayMetrics().density), 0, 0);
            adapterList.addView(title);
            adapterList.addView(text((a.getBoolean("received") ? "Shared with you by someone else. " : "Trained from this device. ")
                    + (a.optBoolean("continued") ? "Continued from an earlier adapter. " : "")
                    + (a.isNull("community") ? "" : "Community rating " + a.optDouble("community") + " out of 5. ")
                    + (a.getBoolean("my_score") ? "You have scored it on your own questions."
                    : a.getBoolean("scored") ? "Scored by the run that trained it." : "Not scored yet."), 13));
            adapterList.addView(small("Try it", v -> {
                trialOf = name;
                trialTurns.clear();
                trialPending.clear();
                trialTitle.setText("Try \"" + name + "\"");
                trialPanel.setVisibility(View.VISIBLE);
                showTrialMode();
            }));
            adapterList.addView(small("Score on my pairs", v -> scoreOnMyPairs(name)));
            adapterList.addView(small("Share score card", v -> shareScoreCard(name)));
            adapterList.addView(small("Save a copy\u2026", v -> {
                savingAdapter = name;
                startActivityForResult(new Intent(Intent.ACTION_CREATE_DOCUMENT).addCategory(Intent.CATEGORY_OPENABLE)
                        .setType("application/zip").putExtra(Intent.EXTRA_TITLE, name + ".zip"), SAVE_ADAPTER);
            }));
            adapterList.addView(small("Check it for flattery", v -> {
                String[] c = connection(sideNote);
                if (c != null) {
                    sideNote.setText("Checking \"" + name + "\" against the unchanged model on questions built on a "
                            + "false statement\u2026");
                    lastSide = "";
                    work.execute(() -> PeerService.bridge(this).callAttr("adapter_flattery", c[0], c[1],
                            c[2].split("[,\\s]+"), Integer.parseInt(c[3]), getFilesDir().getAbsolutePath(), name));
                }
            }));
            adapterList.addView(small("Lead a team with it", v -> teamDo("team", "create", named(name), false)));
            adapterList.addView(small("Train it further", v -> {
                trainFrom = name;
                pairsNote.setText("The next training run continues from \"" + name + "\" and saves a new adapter under "
                        + "the run name below. \"" + name + "\" itself is kept as it is. Choose your pairs and press "
                        + "Start training.");
            }));
            final int helpers = a.optInt("helpers");
            if (helpers > 0 && !a.getBoolean("received")) {
                adapterList.addView(small("Offer it to the " + helpers + " device" + (helpers == 1 ? "" : "s")
                        + " that helped train it", v -> teamDo("share", "open", named(name), false)));
            }
        }
    }

    private TextView label(String s) {
        TextView t = text(s, 13);
        t.setPadding(0, (int) (14 * getResources().getDisplayMetrics().density), 0, 0);
        return t;
    }

    private EditText field(String value, int type) {
        EditText e = new EditText(this);
        e.setInputType(type);
        e.setText(value);
        e.setLayoutParams(new LinearLayout.LayoutParams(ViewGroup.LayoutParams.MATCH_PARENT,
                ViewGroup.LayoutParams.WRAP_CONTENT));
        return e;
    }

    private File importedModels() {
        File dir = new File(getFilesDir(), "models");
        dir.mkdirs();
        return dir;
    }

    /** Models are looked for in the app's own storage: imported ones, and ones copied in over USB. */
    private void refreshModels() {
        work.execute(() -> {
            List<String> names = new ArrayList<>();
            List<String> paths = new ArrayList<>();
            String note = "";
            try {
                File external = getExternalFilesDir("models");
                String[] folders = {importedModels().getAbsolutePath(),
                        external == null ? "" : external.getAbsolutePath()};
                JSONArray found = new JSONArray(PeerService.bridge(this).callAttr("models", (Object) folders).toString());
                for (int i = 0; i < found.length(); i++) {
                    JSONObject m = found.getJSONObject(i);
                    if (m.getBoolean("ok")) {
                        names.add(m.getString("name") + "  (" + m.getString("family") + ", "
                                + m.getInt("layers") + " layers)");
                        paths.add(m.getString("path"));
                    } else {
                        note += m.getString("name") + " cannot be used: " + m.getString("why") + "\n";
                    }
                }
                if (names.isEmpty()) {
                    note += "No model yet. Import a folder holding config.json and the .safetensors files"
                            + (external == null ? "." : ", or copy one into " + external.getAbsolutePath() + ".");
                }
            } catch (Exception e) {
                note = "Could not list models: " + e.getMessage();
                Log.e(TAG, "models", e);
            }
            final String shown = note.trim();
            ui.post(() -> {
                modelPaths.clear();
                modelPaths.addAll(paths);
                modelPicker.setAdapter(new ArrayAdapter<>(this, android.R.layout.simple_spinner_dropdown_item, names));
                int last = paths.indexOf(prefs.getString("model", ""));
                if (last >= 0) {
                    modelPicker.setSelection(last);
                }
                modelNote.setText(shown);
                modelNote.setVisibility(shown.isEmpty() ? View.GONE : View.VISIBLE);
            });
        });
    }

    private void startFromScreen() {
        String code = groupCode.getText().toString().trim();
        int chosen = modelPicker.getSelectedItemPosition();
        if (!understood.isChecked()) {
            status.setText("Tick the box to confirm you have read the warning.");
            return;
        }
        if (chosen < 0 || chosen >= modelPaths.size()) {
            status.setText("Import a model first.");
            return;
        }
        if (code.replace(" ", "").length() < 8) {
            status.setText("The group code needs at least 8 characters (or enter: public).");
            return;
        }
        int size;
        try {
            size = Math.max(1, Integer.parseInt(span.getText().toString().trim()));
        } catch (NumberFormatException e) {
            size = 2;
        }
        String range = layers.getText().toString().trim();
        prefs.edit().putString("group", code).putString("peers", peers.getText().toString().trim())
                .putString("layers", range).putString("span", String.valueOf(size))
                .putBoolean("training", training.isChecked()).putString("model", modelPaths.get(chosen)).apply();
        launch(modelPaths.get(chosen), code, peers.getText().toString().trim(),
                range.isEmpty() ? "auto" : range, size, training.isChecked(), 7700);
    }

    private void launch(String model, String code, String peerList, String range, int size, boolean help, int port) {
        Intent intent = new Intent(this, PeerService.class)
                .putExtra("model", model).putExtra("group", code).putExtra("peers", peerList)
                .putExtra("layers", range).putExtra("span", size).putExtra("training", help).putExtra("port", port);
        if (Build.VERSION.SDK_INT >= 26) {
            startForegroundService(intent);
        } else {
            startService(intent);
        }
    }

    private void stopService() {
        startService(new Intent(this, PeerService.class).setAction(PeerService.ACTION_STOP));
    }

    /** Test builds only: writes what the screen would show to a file a developer can read back. */
    private void autoDump(String what, String said) {
        try {
            com.chaquo.python.PyObject app = PeerService.bridge(this);
            JSONObject out = new JSONObject().put("what", what).put("said", said)
                    .put("at", System.currentTimeMillis())
                    .put("panel", new JSONObject(app.callAttr("team_poll").toString()))
                    .put("ask", app.callAttr("ask_poll").toString())
                    .put("status", app.callAttr("status").toString());
            java.io.File file = new java.io.File(getFilesDir(), "auto-last.json");
            java.io.File part = new java.io.File(getFilesDir(), "auto-last.json.part");
            try (java.io.FileWriter w = new java.io.FileWriter(part)) {
                w.write(out.toString());
            }
            part.renameTo(file);
            Log.i(TAG, "AUTO_DUMP " + what + " " + out.getLong("at"));
        } catch (Exception e) {
            Log.e(TAG, "auto dump", e);
        }
    }

    /** Test builds only: lets a developer start the service from a command instead of the screen. */
    private void automate(Intent intent) {
        boolean debuggable = (getApplicationInfo().flags & ApplicationInfo.FLAG_DEBUGGABLE) != 0;
        if (!debuggable || intent == null) {
            return;
        }
        if (intent.getBooleanExtra("self_check", false)) {
            work.execute(() -> Log.i(TAG, "SELF_CHECK " + PeerService.bridge(this).callAttr("self_check")));
        }
        String model = intent.getStringExtra("auto_model");
        if (intent.getStringExtra("auto_ask") != null) {
            askWith(model, intent.getStringExtra("auto_group"), intent.getStringExtra("auto_peers"),
                    intent.getIntExtra("auto_own", 2), intent.getStringExtra("auto_ask"),
                    intent.getIntExtra("auto_new", 10), 0.0);
        } else if (intent.getStringExtra("auto_try") != null) {
            final String name = intent.getStringExtra("auto_name"), items = intent.getStringExtra("auto_try");
            final Intent in = intent;
            trialOf = name;
            work.execute(() -> {
                try {
                    org.json.JSONArray list = new org.json.JSONArray();
                    for (String q : items.split("\\|")) {
                        trialPending.add(q);
                        org.json.JSONArray m = new org.json.JSONArray().put(new JSONObject().put("role", "user").put("content", q));
                        list.put(new JSONObject().put("without", m).put("with", m));
                    }
                    Log.i(TAG, "TRY_START " + PeerService.bridge(this).callAttr("adapter_try", model,
                            in.getStringExtra("auto_group"), in.getStringExtra("auto_peers").split("[,\\s]+"),
                            in.getIntExtra("auto_own", 2), getFilesDir().getAbsolutePath(), name, list.toString(), 6));
                } catch (Exception e) {
                    Log.e(TAG, "auto try", e);
                }
            });
        } else if (intent.getStringExtra("auto_score") != null) {
            final Intent in = intent;
            work.execute(() -> Log.i(TAG, "SCORE_START " + PeerService.bridge(this).callAttr("adapter_score", model,
                    in.getStringExtra("auto_group"), in.getStringExtra("auto_peers").split("[,\\s]+"),
                    in.getIntExtra("auto_own", 2), getFilesDir().getAbsolutePath(), in.getStringExtra("auto_name"),
                    in.getStringExtra("auto_score"))));
        } else if (intent.getBooleanExtra("auto_poll", false)) {
            work.execute(() -> autoDump("poll", ""));
        } else if (intent.getStringExtra("auto_team") != null) {
            final Intent in = intent;
            work.execute(() -> {
                String said = PeerService.bridge(this).callAttr("panel_do", model,
                        in.getStringExtra("auto_group"), in.getStringExtra("auto_peers").split("[,\\s]+"),
                        in.getIntExtra("auto_own", 2), getFilesDir().getAbsolutePath(),
                        in.getStringExtra("auto_kind") == null ? "team" : in.getStringExtra("auto_kind"),
                        in.getStringExtra("auto_team"),
                        in.getStringExtra("auto_req") == null ? "{}" : in.getStringExtra("auto_req")).toString();
                Log.i(TAG, "TEAM_AUTO [" + said + "]");
                autoDump(in.getStringExtra("auto_team"), said);
            });
        } else if (intent.getStringExtra("auto_import") != null) {
            final Intent in = intent;
            work.execute(() -> Log.i(TAG, "IMPORTED [" + PeerService.bridge(this).callAttr("adapter_import",
                    getFilesDir().getAbsolutePath(), in.getStringExtra("auto_name"), in.getStringExtra("auto_import")) + "]"));
        } else if (intent.getStringExtra("auto_train") != null) {
            trainWith(model, intent.getStringExtra("auto_group"), intent.getStringExtra("auto_peers"),
                    intent.getIntExtra("auto_own", 2), intent.getStringExtra("auto_train"),
                    intent.getStringExtra("auto_name"), 2.0);
        } else if (model != null) {
            launch(intent.getStringExtra("auto_model"), intent.getStringExtra("auto_group"),
                    intent.getStringExtra("auto_peers"), intent.getStringExtra("auto_layers"),
                    intent.getIntExtra("auto_span", 2), intent.getBooleanExtra("auto_training", false),
                    intent.getIntExtra("auto_port", 7700));
        }
    }

    private void tick() {
        work.execute(() -> {
            String line;
            boolean on = false;
            try {
                JSONObject s = new JSONObject(PeerService.bridge(this).callAttr("status").toString());
                String phase = s.getString("phase");
                on = phase.equals("running") || phase.equals("starting");
                if (phase.equals("running")) {
                    JSONArray range = s.getJSONArray("layers");
                    line = "On. Serving layers " + range.getInt(0) + "-" + (range.getInt(1) - 1)
                            + " on port " + s.getInt("port") + ".\n"
                            + "Peers reachable: " + s.getInt("peers") + " (" + s.getInt("serving_peers")
                            + " serving this model)\n"
                            + "Answer steps helped: " + s.getInt("steps") + "\n"
                            + "Training examples helped: " + s.getInt("examples")
                            + (s.getBoolean("training") ? "" : " (training help is off)");
                } else if (phase.equals("starting")) {
                    line = "Starting: joining the group and loading layers…";
                } else if (phase.equals("failed")) {
                    line = "Could not start: " + s.optString("error");
                } else {
                    line = "Off.";
                }
                Log.i(TAG, "STATUS " + s);
            } catch (Exception e) {
                line = "Status unavailable: " + e.getMessage();
                Log.e(TAG, "status", e);
            }
            String reply = null, trainLine = "";
            boolean busyAsking = false, busyTraining = false;
            JSONArray adapters = new JSONArray();
            try {
                JSONObject a = new JSONObject(PeerService.bridge(this).callAttr("ask_poll").toString());
                busyAsking = !a.getBoolean("done");
                reply = a.getString("text");
                if (!a.isNull("error")) {
                    reply = a.getString("error");
                } else if (!a.isNull("stats")) {
                    JSONObject st = a.getJSONObject("stats");
                    reply += "\n\n" + st.getInt("tokens") + " tokens in " + st.getDouble("seconds") + " s through "
                            + st.getInt("peers") + " device" + (st.getInt("peers") == 1 ? "" : "s") + ": "
                            + st.getInt("checked") + " steps checked, " + st.getInt("unchecked") + " not checked"
                            + (st.getInt("blocked") > 0 ? ", " + st.getInt("blocked") + " devices blocked" : "") + "."
                            + (st.isNull("team") ? "" : st.optString("team").equals("no adapter")
                                ? "\nAnswered on your team's devices." : "\nAnswered with your team's adapter, \""
                                + st.optString("team") + "\".");
                } else if (busyAsking && reply.isEmpty()) {
                    reply = "Asking the group\u2026";
                }
                if (!busyAsking && (!a.isNull("stats") || !a.isNull("error"))) {
                    Log.i(TAG, "ASK " + a);
                }
                JSONObject t = new JSONObject(PeerService.bridge(this).callAttr("train_poll",
                        getFilesDir().getAbsolutePath()).toString());
                String state = t.getString("state");
                busyTraining = state.equals("starting") || state.equals("running") || state.equals("stopping")
                        || state.equals("scoring");
                adapters = new JSONArray(PeerService.bridge(this).callAttr("adapters_list",
                        getFilesDir().getAbsolutePath()).toString());
                if (!state.equals("idle")) {
                    trainLine = t.getString("name") + ": "
                            + (state.equals("done") ? "finished" : state.equals("failed") ? "stopped with a problem" : state)
                            + "\nStep " + t.getInt("step") + " of " + t.getInt("total") + " \u00b7 " + t.getInt("pairs")
                            + " pairs" + (t.isNull("loss") ? "" : String.format(" \u00b7 loss %.3f", t.getDouble("loss")))
                            + "\n" + t.getInt("checked") + " pieces of work checked, " + t.getInt("unchecked")
                            + " not checked" + (t.getInt("blocked") > 0 ? ", " + t.getInt("blocked") + " devices blocked" : "")
                            + (t.getInt("skipped") > 0 ? "\n" + t.getInt("skipped") + " lines could not be used" : "")
                            + (t.optJSONArray("score_lines") == null || t.getJSONArray("score_lines").length() == 0
                                ? "" : "\n" + t.getJSONArray("score_lines").join("\n").replace("\"", ""))
                            + (t.isNull("error") ? "" : "\n" + t.getString("error"))
                            + (state.equals("stopped") ? "\nProgress is saved. Start again with the same name to carry on." : "");
                    Log.i(TAG, "TRAIN " + t);
                }
            } catch (Exception e) {
                Log.e(TAG, "poll", e);
            }
            JSONObject teamPanel = null;
            try {
                teamPanel = new JSONObject(PeerService.bridge(this).callAttr("team_poll").toString());
            } catch (Exception e) {
                Log.e(TAG, "team poll", e);
            }
            final JSONObject shownTeam = teamPanel;
            String trialShown = null, sideShown = null;
            boolean sideBusy = false;
            try {
                String polled = PeerService.bridge(this).callAttr("adapter_poll").toString();
                if (!polled.equals("null")) {
                    JSONObject j = new JSONObject(polled);
                    sideBusy = j.getString("state").equals("running");
                    JSONObject result = j.optJSONObject("result");
                    if (j.getString("kind").equals("try") && j.getString("name").equals(trialOf)) {
                        JSONArray answers = result == null ? new JSONArray() : result.optJSONArray("answers");
                        StringBuilder out = new StringBuilder();
                        for (String[] turn : trialTurns) {
                            out.append("Q: ").append(turn[0]).append("\nWithout the adapter: ").append(turn[1])
                                    .append("\nWith \"").append(trialOf).append("\": ").append(turn[2]).append("\n\n");
                        }
                        for (int i = 0; i < trialPending.size(); i++) {
                            JSONObject got = answers != null && i < answers.length() ? answers.getJSONObject(i) : null;
                            out.append("Q: ").append(trialPending.get(i)).append("\nWithout the adapter: ")
                                    .append(got == null ? "\u2026" : got.getString("without"))
                                    .append("\nWith \"").append(trialOf).append("\": ")
                                    .append(got == null ? "\u2026" : got.getString("with")).append("\n\n");
                        }
                        if (!j.isNull("error")) {
                            out.append(j.getString("error"));
                        }
                        if (!sideBusy && answers != null && !trialPending.isEmpty()) {
                            if (trialChat) {               // keep what each side said, for the next message
                                for (int i = 0; i < trialPending.size() && i < answers.length(); i++) {
                                    JSONObject got = answers.getJSONObject(i);
                                    trialTurns.add(new String[]{trialPending.get(i), got.getString("without"),
                                            got.getString("with")});
                                }
                                trialPending.clear();
                            }
                            Log.i(TAG, "TRIAL " + j);
                        }
                        trialShown = out.toString().trim();
                    } else if ((j.getString("kind").equals("score") || j.getString("kind").equals("flattery"))
                            && !sideBusy) {
                        StringBuilder out = new StringBuilder();
                        if (!j.isNull("error")) {
                            out.append(j.getString("error"));
                        } else if (result != null) {
                            JSONArray lines = result.getJSONArray("lines");
                            for (int i = 0; i < lines.length(); i++) {
                                out.append(lines.getString(i)).append("\n");
                            }
                        }
                        sideShown = out.toString().trim();
                        Log.i(TAG, "SCORED " + j);
                    }
                }
            } catch (Exception e) {
                Log.e(TAG, "adapter poll", e);
            }
            final String shownTrial = trialShown, shownSide = sideShown;
            final boolean isSideBusy = sideBusy;
            final String shown = line, shownReply = reply, shownTrain = trainLine;
            final boolean isOn = on, isAsking = busyAsking, isTraining = busyTraining;
            final JSONArray list = adapters;
            ui.post(() -> {
                running = isOn;
                asking = isAsking;
                training_on = isTraining;
                status.setText(shown);
                toggle.setText(isOn ? "Turn the p2p service off" : "Turn the p2p service on");
                askButton.setText(isAsking ? "Stop" : "Ask");
                trainButton.setText(isTraining ? "Stop training" : "Start training");
                if (shownReply != null && (isAsking || !shownReply.isEmpty())) {
                    answer.setText(shownReply);
                }
                if (!shownTrain.isEmpty()) {
                    trainStatus.setText(shownTrain);
                }
                trialBusy = isSideBusy;
                if (shownTrial != null && !shownTrial.isEmpty()) {
                    trialOut.setText(shownTrial);
                }
                if (shownSide != null && !shownSide.isEmpty() && !shownSide.equals(lastSide)) {
                    lastSide = shownSide;
                    sideNote.setText(shownSide);
                }
                if (trialOf != null) {
                    trialButton.setText(isSideBusy ? "Stop" : trialChat ? "Send" : "Try these questions");
                }
                try {
                    showAdapters(list);
                    if (shownTeam != null) {
                        showTeam(shownTeam);
                    }
                } catch (Exception e) {
                    Log.e(TAG, "adapters", e);
                }
            });
        });
        ui.postDelayed(this::tick, 1000);
    }

    @Override
    protected void onActivityResult(int request, int result, Intent data) {
        super.onActivityResult(request, result, data);
        if (result != RESULT_OK || data == null || data.getData() == null) {
            return;
        }
        if (request == PICK_PAIRS) {
            Uri source = data.getData();
            work.execute(() -> {
                String note;
                try {
                    File target = new File(getCacheDir(), "pairs.jsonl");
                    copy(getContentResolver().openInputStream(source), new FileOutputStream(target));
                    pairsFile = target;
                    note = "Pairs file ready (" + (target.length() / 1024 + 1) + " KB).";
                } catch (Exception e) {
                    note = "Could not read that file: " + e.getMessage();
                }
                final String shown = note;
                ui.post(() -> pairsNote.setText(shown));
            });
            return;
        }
        if (request == PICK_SHARED) {
            Uri source = data.getData();
            work.execute(() -> {
                String note;
                try {
                    File zip = new File(getCacheDir(), "shared.zip");
                    copy(getContentResolver().openInputStream(source), new FileOutputStream(zip));
                    String name = "shared-" + Long.toString(System.currentTimeMillis() / 1000, 36);
                    String refused = PeerService.bridge(this).callAttr("adapter_import",
                            getFilesDir().getAbsolutePath(), name, zip.getAbsolutePath()).toString();
                    zip.delete();
                    note = refused.isEmpty() ? "Added as \"" + name + "\". Try it and score it before deciding to use it."
                            : "Could not add it: " + refused;
                } catch (Exception e) {
                    note = "Could not read that file: " + e.getMessage();
                }
                final String shown = note;
                ui.post(() -> sideNote.setText(shown));
            });
            return;
        }
        if (request == SAVE_SCORES) {
            Uri target = data.getData();
            work.execute(() -> {
                String note;
                try {
                    File page = new File(getCacheDir(), "model-adapter-score.html");
                    PeerService.bridge(this).callAttr("score_document", getFilesDir().getAbsolutePath(),
                            page.getAbsolutePath());
                    copy(new java.io.FileInputStream(page), getContentResolver().openOutputStream(target));
                    page.delete();
                    note = "Saved Model Adapter Score. Open it in a browser; turn the phone sideways to read it.";
                } catch (Exception e) {
                    note = "Could not save: " + e.getMessage();
                }
                final String shown = note;
                ui.post(() -> sideNote.setText(shown));
            });
            return;
        }
        if (request == SAVE_ADAPTER) {
            Uri target = data.getData();
            final String name = savingAdapter;
            work.execute(() -> {
                String note;
                try {
                    File zip = new File(getCacheDir(), name + ".zip");
                    PeerService.bridge(this).callAttr("adapter_zip", getFilesDir().getAbsolutePath(), name,
                            zip.getAbsolutePath());
                    copy(new java.io.FileInputStream(zip), getContentResolver().openOutputStream(target));
                    zip.delete();
                    note = "Saved a copy of \"" + name + "\". Read READ-ME-FIRST.txt inside before using it.";
                } catch (Exception e) {
                    note = "Could not save: " + e.getMessage();
                }
                final String shown = note;
                ui.post(() -> trainStatus.setText(shown));
            });
            return;
        }
        if (request != PICK_FOLDER) {
            return;
        }
        Uri tree = data.getData();
        importButton.setEnabled(false);
        work.execute(() -> {
            String outcome;
            try {
                outcome = copyModel(tree);
            } catch (Exception e) {
                outcome = "Import failed: " + e.getMessage();
                Log.e(TAG, "import", e);
            }
            final String shown = outcome;
            ui.post(() -> {
                importButton.setEnabled(true);
                status.setText(shown);
                refreshModels();
            });
        });
    }

    private static void copy(InputStream in, OutputStream out) throws Exception {
        try (InputStream from = in; OutputStream to = out) {
            byte[] buffer = new byte[1 << 16];
            int n;
            while ((n = from.read(buffer)) > 0) {
                to.write(buffer, 0, n);
            }
        }
    }

    /** Copies config.json and the weight files from the chosen folder into the app's own storage. */
    private String copyModel(Uri tree) throws Exception {
        ContentResolver resolver = getContentResolver();
        String rootId = DocumentsContract.getTreeDocumentId(tree);
        String name = rootId.replaceAll(".*[:/]", "").replaceAll("[^A-Za-z0-9._-]", "_");
        File target = new File(importedModels(), name.isEmpty() ? "model" : name);
        target.mkdirs();
        Uri children = DocumentsContract.buildChildDocumentsUriUsingTree(tree, rootId);
        int copied = 0;
        try (Cursor c = resolver.query(children, new String[]{DocumentsContract.Document.COLUMN_DOCUMENT_ID,
                DocumentsContract.Document.COLUMN_DISPLAY_NAME}, null, null, null)) {
            while (c != null && c.moveToNext()) {
                String file = c.getString(1);
                if (file == null || !(file.endsWith(".safetensors") || file.endsWith(".json")) || file.contains("/")) {
                    continue;
                }
                final String progress = "Copying " + file + "…";
                ui.post(() -> status.setText(progress));
                Uri doc = DocumentsContract.buildDocumentUriUsingTree(tree, c.getString(0));
                try (InputStream in = resolver.openInputStream(doc);
                     OutputStream out = new FileOutputStream(new File(target, file))) {
                    byte[] buffer = new byte[1 << 20];
                    int n;
                    while ((n = in.read(buffer)) > 0) {
                        out.write(buffer, 0, n);
                    }
                }
                copied++;
            }
        }
        return copied == 0 ? "That folder has no config.json or .safetensors files."
                : "Imported " + copied + " files as \"" + target.getName() + "\".";
    }
}
