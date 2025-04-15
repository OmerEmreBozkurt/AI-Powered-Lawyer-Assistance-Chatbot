import os
import json
import numpy as np
import google.generativeai as genai
import docx
import re
import textwrap
from dotenv import load_dotenv

# Ortam değişkenlerini yükle
load_dotenv()


class KanunRAGChatbot:
    def __init__(self, document_path="docs/Yasa1.docx"):
        """
        Genel kanunlar için RAG Chatbot’u başlatır.
        """
        self.gemini_key = os.getenv('GEMINI_KEY')
        if not self.gemini_key:
            raise ValueError("Please set GEMINI_KEY in your .env file")

        genai.configure(api_key=self.gemini_key)
        self.model = genai.GenerativeModel("gemini-1.5-flash")
        self.embedding_model = "models/embedding-001"

        self.document_path = document_path
        self.kanun_data = self._extract_kanun_data()
        self.embeddings_df = self._generate_embeddings()

    def _extract_kanun_data(self):
        """Docx dosyasından kanun bilgilerini ve maddeleri çıkarır."""
        doc = docx.Document(self.document_path)
        kanun_data = {
            "kanun_adi": "",
            "kanun_numarasi": "",
            "yayim_tarihi": "",
            "resmi_gazete": "",
            "maddeler": [],
            "gecici_maddeler": [],
            "ekler": []
        }

        current_madde = None
        current_content = []
        section = "meta"  # meta, maddeler, gecici_maddeler, ekler

        for para in doc.paragraphs:
            text = para.text.strip()
            if not text:
                continue

            # Kanun meta verilerini çıkar
            if section == "meta":
                if "Kanun Numarası:" in text:
                    kanun_data["kanun_numarasi"] = text.split(":")[1].strip()
                elif text.startswith("**") and text.endswith("**"):
                    kanun_data["kanun_adi"] = text.strip("**").strip()
                elif "Yayımlandığı Resmî Gazete:" in text:
                    kanun_data["resmi_gazete"] = text.split(":", 1)[1].strip()
                elif "Yayımlandığı Düstur:" in text:
                    kanun_data["yayim_tarihi"] = text.split(":", 1)[1].strip()

            # Maddeleri tespit et
            if text.startswith("MADDE ") and "(Ek:" not in text:
                if current_madde:
                    if section == "maddeler":
                        kanun_data["maddeler"].append({
                            "id": f"{kanun_data['kanun_numarasi']}-MADDE-{current_madde}",
                            "title": f"Madde {current_madde}",
                            "content": " ".join(current_content).strip(),
                            "keywords": self._extract_keywords(" ".join(current_content))
                        })
                    elif section == "gecici_maddeler":
                        kanun_data["gecici_maddeler"].append({
                            "id": f"{kanun_data['kanun_numarasi']}-GECICI-MADDE-{current_madde}",
                            "title": f"Geçici Madde {current_madde}",
                            "content": " ".join(current_content).strip(),
                            "keywords": self._extract_keywords(" ".join(current_content))
                        })
                match = re.match(r"MADDE (\d+)", text)
                if match:
                    current_madde = match.group(1)
                    current_content = [text.replace(f"MADDE {current_madde}", "").strip()]
                    section = "maddeler"

            # Geçici maddeleri tespit et
            elif text.startswith("GEÇİCİ MADDE "):
                if current_madde:
                    kanun_data["maddeler"].append({
                        "id": f"{kanun_data['kanun_numarasi']}-MADDE-{current_madde}",
                        "title": f"Madde {current_madde}",
                        "content": " ".join(current_content).strip(),
                        "keywords": self._extract_keywords(" ".join(current_content))
                    })
                match = re.match(r"GEÇİCİ MADDE (\d+)", text)
                if match:
                    current_madde = match.group(1)
                    current_content = [text.replace(f"GEÇİCİ MADDE {current_madde}", "").strip()]
                    section = "gecici_maddeler"

            # Ekler veya diğer bölümler
            elif text.startswith("**") and section != "meta":
                if current_madde:
                    if section == "maddeler":
                        kanun_data["maddeler"].append({
                            "id": f"{kanun_data['kanun_numarasi']}-MADDE-{current_madde}",
                            "title": f"Madde {current_madde}",
                            "content": " ".join(current_content).strip(),
                            "keywords": self._extract_keywords(" ".join(current_content))
                        })
                    elif section == "gecici_maddeler":
                        kanun_data["gecici_maddeler"].append({
                            "id": f"{kanun_data['kanun_numarasi']}-GECICI-MADDE-{current_madde}",
                            "title": f"Geçici Madde {current_madde}",
                            "content": " ".join(current_content).strip(),
                            "keywords": self._extract_keywords(" ".join(current_content))
                        })
                current_madde = None
                current_content = []
                section = "ekler"
                kanun_data["ekler"].append({
                    "title": text.strip("**").strip(),
                    "content": ""
                })

            # İçeriği ekle
            elif current_madde:
                current_content.append(text)
            elif section == "ekler" and kanun_data["ekler"]:
                kanun_data["ekler"][-1]["content"] += " " + text

        # Son maddeyi kaydet
        if current_madde:
            if section == "maddeler":
                kanun_data["maddeler"].append({
                    "id": f"{kanun_data['kanun_numarasi']}-MADDE-{current_madde}",
                    "title": f"Madde {current_madde}",
                    "content": " ".join(current_content).strip(),
                    "keywords": self._extract_keywords(" ".join(current_content))
                })
            elif section == "gecici_maddeler":
                kanun_data["gecici_maddeler"].append({
                    "id": f"{kanun_data['kanun_numarasi']}-GECICI-MADDE-{current_madde}",
                    "title": f"Geçici Madde {current_madde}",
                    "content": " ".join(current_content).strip(),
                    "keywords": self._extract_keywords(" ".join(current_content))
                })

        return kanun_data

    def _extract_keywords(self, text):
        """Basit anahtar kelime çıkarma."""
        common_keywords = [
            "tüketici", "sözleşme", "madde", "kanun", "cayma", "hakkı",
            "sorumluluk", "ayıp", "kredi", "faiz", "satıcı", "sağlayıcı"
        ]
        words = text.lower().split()
        return [word for word in common_keywords if word in words]

    def _generate_embedding(self, text):
        """Metin için gömme vektörü üretir."""
        try:
            response = genai.embed_content(
                model=self.embedding_model,
                content=text,
                task_type="RETRIEVAL_DOCUMENT"
            )
            return response["embedding"]
        except Exception as e:
            print(f"Gömme üretirken hata: {e}")
            return [0.0] * 768  # Varsayılan sıfır vektörü

    def _generate_embeddings(self):
        """Maddeler için gömme vektörleri üretir veya yükler."""
        embeddings_file = "kanun_embeddings.json"

        # Mevcut JSON dosyasını yükle
        existing_data = None
        if os.path.exists(embeddings_file):
            try:
                with open(embeddings_file, "r", encoding="utf-8") as f:
                    existing_data = json.load(f)
                print("Mevcut kanun_embeddings.json yüklendi.")
            except Exception as e:
                print(f"JSON yüklenirken hata: {e}. Yeni gömmeler üretilecek.")

        # Yeni veriyi mevcut veriyle birleştir
        if existing_data:
            # Mevcut maddeleri ID’ye göre eşleştir
            existing_maddeler = {m["id"]: m for m in existing_data.get("maddeler", [])}
            existing_gecici = {m["id"]: m for m in existing_data.get("gecici_maddeler", [])}
            existing_ekler = {(e.get("title") or ""): e for e in existing_data.get("ekler", [])}

            # Yeni maddeler için gömme üret veya mevcutları kullan
            for madde in self.kanun_data["maddeler"]:
                if madde["id"] in existing_maddeler and "embedding" in existing_maddeler[madde["id"]]:
                    madde["embedding"] = existing_maddeler[madde["id"]]["embedding"]
                else:
                    madde["embedding"] = self._generate_embedding(madde["content"])

            for madde in self.kanun_data["gecici_maddeler"]:
                if madde["id"] in existing_gecici and "embedding" in existing_gecici[madde["id"]]:
                    madde["embedding"] = existing_gecici[madde["id"]]["embedding"]
                else:
                    madde["embedding"] = self._generate_embedding(madde["content"])

            for ek in self.kanun_data["ekler"]:
                ek_title = ek.get("title", "")
                if ek_title in existing_ekler and "embedding" in existing_ekler[ek_title] and ek["content"]:
                    ek["embedding"] = existing_ekler[ek_title]["embedding"]
                else:
                    ek["embedding"] = self._generate_embedding(ek["content"]) if ek["content"] else [0.0] * 768
        else:
            # JSON yoksa tüm gömmeleri üret
            for madde in self.kanun_data["maddeler"]:
                madde["embedding"] = self._generate_embedding(madde["content"])
            for madde in self.kanun_data["gecici_maddeler"]:
                madde["embedding"] = self._generate_embedding(madde["content"])
            for ek in self.kanun_data["ekler"]:
                ek["embedding"] = self._generate_embedding(ek["content"]) if ek["content"] else [0.0] * 768

        # JSON’a kaydet
        try:
            with open(embeddings_file, "w", encoding="utf-8") as f:
                json.dump(self.kanun_data, f, ensure_ascii=False, indent=2)
            print(f"{len(self.kanun_data['maddeler'])} madde, "
                  f"{len(self.kanun_data['gecici_maddeler'])} geçici madde ve "
                  f"{len(self.kanun_data['ekler'])} ek işlendi.")
        except Exception as e:
            print(f"JSON kaydederken hata: {e}")

        return self.kanun_data

    def find_best_passage(self, query, top_k=2):
        """En uygun maddeleri bulur."""
        query_embedding = genai.embed_content(
            model=self.embedding_model,
            content=query,
            task_type="RETRIEVAL_QUERY"
        )["embedding"]

        all_maddeler = self.kanun_data["maddeler"] + self.kanun_data["gecici_maddeler"]
        embeddings = np.array([m.get("embedding", [0.0] * 768) for m in all_maddeler])
        dot_products = np.dot(embeddings, query_embedding)
        top_indices = np.argsort(dot_products)[::-1][:top_k]

        return [all_maddeler[i] for i in top_indices]

    def generate_response(self, query):
        """Soruya yanıt üretir."""
        maddeler = self.find_best_passage(query)

        prompt = textwrap.dedent(f"""You are a helpful legal bot that answers questions based on Turkish laws.
        Respond in a complete, conversational sentence in Turkish. Explain complicated concepts simply.

        SORU: '{query}'
        İLGİLİ MADDELER:
        """)
        for i, madde in enumerate(maddeler):
            prompt += f"{i + 1}. {self.kanun_data['kanun_adi']} {madde['title']} - {madde['content']}\n"

        prompt += f"""
        Bu maddelere dayanarak, soruya kısa, anlaşılır ve doğru bir yanıt ver. Yanıtın sonunda ilgili maddeleri kaynak olarak belirt.
        Kanun: {self.kanun_data['kanun_adi']} ({self.kanun_data['kanun_numarasi']})
        """

        try:
            response = self.model.generate_content(
                prompt,
                generation_config={
                    "max_output_tokens": 500,
                    "temperature": 0.1
                }
            )
            return response.text
        except Exception as e:
            return f"Yanıt üretirken hata: {e}"


def main():
    print("🤖 Lawyer Assistance Chatbot’una hoş geldiniz! 📄")
    print("Çıkmak için 'quit' yazın.\n")

    chatbot = KanunRAGChatbot()

    while True:
        query = input("Soru: ").strip()

        if query.lower() in ['quit', 'exit', 'bye']:
            print("Görüşmek üzere! 👋")
            break

        if not query:
            continue

        print("\n🤖 Düşünüyorum...\n")
        response = chatbot.generate_response(query)
        print("Yanıt:", response, "\n")


if __name__ == "__main__":
    main()