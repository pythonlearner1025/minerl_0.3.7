// Add these imports at the top of MalmoEnvServer.java
import java.util.UUID;
import java.nio.charset.StandardCharsets;
import java.lang.reflect.Field;
import net.minecraft.util.Session;

// Add this method after line 674 (after the interact() method declaration)
private void setUsernameWithUUID(String username) {
    try {
        Minecraft mc = Minecraft.getMinecraft();
        Session currentSession = mc.getSession();

        // Generate UUID using Minecraft's offline algorithm
        UUID offlineUUID = UUID.nameUUIDFromBytes(("OfflinePlayer:" + username).getBytes(StandardCharsets.UTF_8));
        String playerID = offlineUUID.toString().replaceAll("-", "");

        System.out.println("[MineRL UUID Fix] Generated UUID for " + username + ": " + playerID);
        TCPUtils.Log(Level.INFO, "[MineRL UUID Fix] Generated UUID for " + username + ": " + playerID);

        // Create new session with generated UUID
        Session newSession = new Session(username, playerID, currentSession.getToken(), "mojang");

        // Use reflection to inject into Minecraft singleton
        Field sessionField = Minecraft.class.getDeclaredField("session");
        sessionField.setAccessible(true);
        sessionField.set(mc, newSession);

        System.out.println("[MineRL UUID Fix] Session updated successfully");
    } catch (Exception e) {
        System.out.println("[MineRL UUID Fix] ERROR: " + e.getMessage());
        e.printStackTrace();
    }
}

// Modify the interact() method around line 722 to call setUsernameWithUUID():
private void interact(String command, Socket socket) throws IOException {
    lock.lock();
    try {
        String token = command.substring(interactTagLength, command.length() - (interactTagLength + 1));
        String[] tokenSplits = token.split(":");
        String ip = tokenSplits[0];
        String port = tokenSplits[1];

        // ADDED: Generate UUID before connecting
        String username = "Player" + System.currentTimeMillis() % 10000;
        setUsernameWithUUID(username);

        final ServerData sd = new ServerData("agent server", ip + ":" + port, false);
        // ... rest of method
    }
}
