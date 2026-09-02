import { initializeApp, type FirebaseOptions } from "firebase/app";
import { getAuth, GoogleAuthProvider, type Auth } from "firebase/auth";
import { getFirestore, type Firestore } from "firebase/firestore";
import { getStorage, type FirebaseStorage } from "firebase/storage";

const firebaseConfig: FirebaseOptions = {
  apiKey: import.meta.env.VITE_FIREBASE_API_KEY?.trim(),
  authDomain: import.meta.env.VITE_FIREBASE_AUTH_DOMAIN?.trim(),
  projectId: import.meta.env.VITE_FIREBASE_PROJECT_ID?.trim(),
  storageBucket: import.meta.env.VITE_FIREBASE_STORAGE_BUCKET?.trim(),
  messagingSenderId: import.meta.env.VITE_FIREBASE_MESSAGING_SENDER_ID?.trim(),
  appId: import.meta.env.VITE_FIREBASE_APP_ID?.trim(),
};

export const firebaseConfigured = Object.values(firebaseConfig).every(
  (value) => typeof value === "string" && value.length > 0,
);
export const firebaseProjectId = firebaseConfig.projectId ?? "";

type FirebaseServices = {
  auth: Auth;
  db: Firestore;
  storage: FirebaseStorage;
  googleProvider: GoogleAuthProvider;
};

let services: FirebaseServices | null = null;

if (firebaseConfigured) {
  const firebaseApp = initializeApp(firebaseConfig);
  const googleProvider = new GoogleAuthProvider();
  googleProvider.setCustomParameters({ prompt: "select_account" });
  services = {
    auth: getAuth(firebaseApp),
    db: getFirestore(firebaseApp),
    storage: getStorage(firebaseApp),
    googleProvider,
  };
}

export const getFirebaseServices = () => {
  if (!services) {
    throw new Error("Firebase 환경변수가 설정되지 않았습니다.");
  }
  return services;
};

