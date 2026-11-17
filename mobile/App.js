import React, { useEffect, useRef, useState } from "react";
import {
  ActivityIndicator,
  FlatList,
  KeyboardAvoidingView,
  Platform,
  SafeAreaView,
  StatusBar,
  StyleSheet,
  Text,
  TextInput,
  TouchableOpacity,
  View,
} from "react-native";
import * as SecureStore from "expo-secure-store";

const API_URL = (process.env.EXPO_PUBLIC_API_URL || "").replace(/\/+$/, "");
const TOKEN_KEY = "personal_ai_access_token";
// The backend accepts at most 30 messages per request, so only the latest ones are sent.
const MAX_HISTORY = 20;
// Render's free plan can take up to a minute to wake up after being idle.
const REQUEST_TIMEOUT_MS = 90000;

async function fetchJson(path, options = {}) {
  const controller = new AbortController();
  const timer = setTimeout(() => controller.abort(), REQUEST_TIMEOUT_MS);
  try {
    const response = await fetch(`${API_URL}${path}`, { ...options, signal: controller.signal });
    let data = {};
    try {
      data = await response.json();
    } catch (e) {
      data = {};
    }
    return { response, data };
  } catch (e) {
    if (e && e.name === "AbortError") {
      throw new Error("The server took too long to answer. If it was asleep it needs about a minute to wake up. Please try again.");
    }
    throw new Error("Could not reach the server. Check your internet connection and the backend URL.");
  } finally {
    clearTimeout(timer);
  }
}

function errorText(data, fallback) {
  return data && typeof data.detail === "string" ? data.detail : fallback;
}

function trimHistory(list) {
  const recent = list.slice(-MAX_HISTORY).map(({ role, content }) => ({ role, content }));
  while (recent.length > 1 && recent[0].role !== "user") recent.shift();
  return recent;
}

export default function App() {
  const [token, setToken] = useState(null);
  const [password, setPassword] = useState("");
  const [checkingSession, setCheckingSession] = useState(true);
  const [loggingIn, setLoggingIn] = useState(false);
  const [messages, setMessages] = useState([
    { id: "welcome", role: "assistant", content: "Hey! I'm your personal AI assistant. Sign in to start chatting." },
  ]);
  const [draft, setDraft] = useState("");
  const [sending, setSending] = useState(false);
  const [error, setError] = useState("");
  const listRef = useRef(null);

  useEffect(() => {
    // Wake the backend up early so the first message is not slow.
    if (API_URL) fetch(`${API_URL}/health`).catch(() => {});
    (async () => {
      try {
        const saved = await SecureStore.getItemAsync(TOKEN_KEY);
        if (saved) setToken(saved);
      } catch (e) {
        setError("Could not read the saved sign-in session.");
      } finally {
        setCheckingSession(false);
      }
    })();
  }, []);

  async function signIn() {
    setError("");
    if (!API_URL) {
      setError("Backend URL is missing. Set EXPO_PUBLIC_API_URL in mobile/.env and rebuild.");
      return;
    }
    if (!password.trim()) {
      setError("Enter your private owner password.");
      return;
    }
    setLoggingIn(true);
    try {
      const { response, data } = await fetchJson("/auth/login", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ password }),
      });
      if (!response.ok) throw new Error(errorText(data, "Sign-in failed."));
      await SecureStore.setItemAsync(TOKEN_KEY, data.access_token);
      setToken(data.access_token);
      setPassword("");
      setMessages([{ id: "welcome", role: "assistant", content: "You're connected. What would you like to work on?" }]);
    } catch (e) {
      setError(e.message || "Could not connect to the assistant.");
    } finally {
      setLoggingIn(false);
    }
  }

  async function signOut() {
    await SecureStore.deleteItemAsync(TOKEN_KEY);
    setToken(null);
    setMessages([{ id: "welcome", role: "assistant", content: "You have signed out." }]);
  }

  function newChat() {
    if (sending) return;
    setError("");
    setMessages([{ id: "welcome", role: "assistant", content: "New chat started. What would you like to work on?" }]);
  }

  async function sendMessage() {
    const text = draft.trim();
    if (!text || sending || !token) return;
    const previous = messages;
    const nextMessages = [
      ...messages.filter(m => m.id !== "welcome"),
      { id: `${Date.now()}`, role: "user", content: text },
    ];
    setDraft("");
    setError("");
    setMessages(nextMessages);
    setSending(true);
    let sessionExpired = false;
    try {
      const { response, data } = await fetchJson("/chat", {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
          "Authorization": `Bearer ${token}`,
        },
        body: JSON.stringify({ messages: trimHistory(nextMessages) }),
      });
      if (response.status === 401) {
        sessionExpired = true;
        await signOut();
        throw new Error("Your session expired. Please sign in again.");
      }
      if (!response.ok) throw new Error(errorText(data, "The assistant could not answer."));
      setMessages([
        ...nextMessages,
        { id: `${Date.now()}-reply`, role: "assistant", content: data.reply, skills: data.skills_used || [] },
      ]);
    } catch (e) {
      if (!sessionExpired) {
        // Roll back so a failed message is not left in the history, and give the text back to edit.
        setMessages(previous);
        setDraft(text);
      }
      setError(e.message || "Network error. Check your internet and backend URL.");
    } finally {
      setSending(false);
    }
  }

  if (checkingSession) {
    return (
      <SafeAreaView style={styles.center}>
        <ActivityIndicator size="large" />
        <Text style={styles.muted}>Starting your assistant…</Text>
      </SafeAreaView>
    );
  }

  if (!token) {
    return (
      <SafeAreaView style={styles.screen}>
        <StatusBar barStyle="dark-content" />
        <View style={styles.loginCard}>
          <Text style={styles.eyebrow}>PRIVATE • SINGLE USER</Text>
          <Text style={styles.title}>Personal AI Assistant</Text>
          <Text style={styles.subtitle}>Sign in with the owner password configured on your backend.</Text>
          <TextInput
            value={password}
            onChangeText={setPassword}
            placeholder="Owner password"
            placeholderTextColor="#858585"
            secureTextEntry
            autoCapitalize="none"
            style={styles.input}
            onSubmitEditing={signIn}
          />
          {error ? <Text style={styles.error}>{error}</Text> : null}
          <TouchableOpacity style={styles.primaryButton} onPress={signIn} disabled={loggingIn}>
            {loggingIn ? <ActivityIndicator color="#fff" /> : <Text style={styles.primaryButtonText}>Sign in</Text>}
          </TouchableOpacity>
          <Text style={styles.smallNote}>Your AI provider keys stay on your backend, not in this app.</Text>
        </View>
      </SafeAreaView>
    );
  }

  return (
    <SafeAreaView style={styles.screen}>
      <StatusBar barStyle="dark-content" />
      <KeyboardAvoidingView
        style={styles.flex}
        behavior="padding"
        keyboardVerticalOffset={Platform.OS === "ios" ? 8 : 0}
      >
        <View style={styles.header}>
          <View>
            <Text style={styles.eyebrow}>PRIVATE ASSISTANT</Text>
            <Text style={styles.headerTitle}>AI Assistant</Text>
          </View>
          <View style={styles.headerButtons}>
            <TouchableOpacity onPress={newChat} style={styles.signOutButton}>
              <Text style={styles.signOutText}>New chat</Text>
            </TouchableOpacity>
            <TouchableOpacity onPress={signOut} style={styles.signOutButton}>
              <Text style={styles.signOutText}>Sign out</Text>
            </TouchableOpacity>
          </View>
        </View>

        <FlatList
          ref={listRef}
          data={messages}
          keyExtractor={item => item.id}
          contentContainerStyle={styles.messageList}
          onContentSizeChange={() => listRef.current?.scrollToEnd({ animated: true })}
          renderItem={({ item }) => (
            <View style={[styles.bubble, item.role === "user" ? styles.userBubble : styles.assistantBubble]}>
              <Text style={styles.roleLabel}>{item.role === "user" ? "YOU" : "ASSISTANT"}</Text>
              <Text style={styles.messageText} selectable>{item.content}</Text>
              {item.skills && item.skills.length > 0 ? (
                <Text style={styles.skillsLabel}>Used: {item.skills.join(", ")}</Text>
              ) : null}
            </View>
          )}
          ListFooterComponent={sending ? (
            <View style={[styles.bubble, styles.assistantBubble, styles.typing]}>
              <ActivityIndicator size="small" />
              <Text style={styles.muted}>Thinking…</Text>
            </View>
          ) : null}
        />

        {error ? <Text style={styles.errorInline}>{error}</Text> : null}
        <View style={styles.composer}>
          <TextInput
            value={draft}
            onChangeText={setDraft}
            placeholder="Message your assistant…"
            placeholderTextColor="#858585"
            multiline
            style={styles.composerInput}
            editable={!sending}
          />
          <TouchableOpacity style={[styles.sendButton, (!draft.trim() || sending) && styles.disabledButton]} onPress={sendMessage} disabled={!draft.trim() || sending}>
            <Text style={styles.sendText}>Send</Text>
          </TouchableOpacity>
        </View>
      </KeyboardAvoidingView>
    </SafeAreaView>
  );
}

const styles = StyleSheet.create({
  flex: { flex: 1 },
  screen: { flex: 1, backgroundColor: "#F5F6F8" },
  center: { flex: 1, alignItems: "center", justifyContent: "center", backgroundColor: "#F5F6F8", gap: 12 },
  loginCard: { flex: 1, justifyContent: "center", padding: 26 },
  eyebrow: { color: "#667085", fontSize: 11, fontWeight: "700", letterSpacing: 1.5 },
  title: { color: "#151923", fontSize: 30, fontWeight: "800", marginTop: 10 },
  subtitle: { color: "#667085", fontSize: 15, lineHeight: 22, marginTop: 10, marginBottom: 24 },
  input: { backgroundColor: "#FFFFFF", borderWidth: 1, borderColor: "#D8DCE3", borderRadius: 14, padding: 16, color: "#151923", fontSize: 16 },
  primaryButton: { backgroundColor: "#202938", borderRadius: 14, padding: 16, alignItems: "center", marginTop: 14 },
  primaryButtonText: { color: "#FFFFFF", fontWeight: "700", fontSize: 16 },
  smallNote: { color: "#7B8494", fontSize: 12, lineHeight: 18, marginTop: 18, textAlign: "center" },
  header: { paddingHorizontal: 18, paddingVertical: 14, borderBottomWidth: 1, borderBottomColor: "#E3E6EB", flexDirection: "row", justifyContent: "space-between", alignItems: "center", backgroundColor: "#FFFFFF" },
  headerTitle: { color: "#151923", fontSize: 21, fontWeight: "800", marginTop: 3 },
  headerButtons: { flexDirection: "row", gap: 8 },
  skillsLabel: { color: "#667085", fontSize: 11, marginTop: 8 },
  signOutButton: { paddingHorizontal: 12, paddingVertical: 9, borderRadius: 10, backgroundColor: "#EEF0F4" },
  signOutText: { color: "#344054", fontWeight: "600" },
  messageList: { padding: 14, gap: 10 },
  bubble: { maxWidth: "88%", borderRadius: 16, padding: 13, marginVertical: 4 },
  userBubble: { alignSelf: "flex-end", backgroundColor: "#DCEAFE" },
  assistantBubble: { alignSelf: "flex-start", backgroundColor: "#FFFFFF", borderWidth: 1, borderColor: "#E5E7EB" },
  roleLabel: { color: "#667085", fontSize: 9, fontWeight: "800", letterSpacing: 1, marginBottom: 6 },
  messageText: { color: "#202938", fontSize: 15, lineHeight: 22 },
  typing: { flexDirection: "row", alignItems: "center", gap: 10 },
  muted: { color: "#667085", fontSize: 13 },
  error: { color: "#B42318", marginTop: 10, lineHeight: 19 },
  errorInline: { color: "#B42318", paddingHorizontal: 14, paddingBottom: 6, fontSize: 12 },
  composer: { flexDirection: "row", alignItems: "flex-end", padding: 10, gap: 8, borderTopWidth: 1, borderTopColor: "#E3E6EB", backgroundColor: "#FFFFFF" },
  composerInput: { flex: 1, maxHeight: 120, minHeight: 44, backgroundColor: "#F2F4F7", borderRadius: 14, paddingHorizontal: 13, paddingVertical: 11, color: "#202938", fontSize: 15 },
  sendButton: { backgroundColor: "#202938", paddingHorizontal: 17, paddingVertical: 13, borderRadius: 13 },
  disabledButton: { opacity: 0.45 },
  sendText: { color: "#FFFFFF", fontWeight: "700" }
});
