using StableDiffusionSharp;
using System;
using System.Diagnostics;
using System.Drawing;
using System.IO;
using System.Threading.Tasks;
using System.Windows.Forms;

namespace StableDiffusionSharpGUI
{
    public partial class MainForm : Form
    {
        private readonly SDHelper helper = new SDHelper();
        public MainForm()
        {
            InitializeComponent();
        }

        private void Button_LoadModel_Click(object sender, EventArgs e)
        {
            if (Button_LoadModel.Text == "Load Model")
            {
                if (!helper.IsInitialized)
                {
                    string modelPath = TextBox_ModelPath.Text;
                    string vaePath = TextBox_VaePath.Text;
                    string loraModelDir = TextBox_LoraModelDir.Text;
                    bool keepVaeOnCpu = CheckBox_CpuVae.Checked;
                    bool vaeTiling = CheckBox_VaeTiling.Checked;
                    if (!File.Exists(modelPath))
                    {
                        Button_LoadModel.Enabled = true;
                        MessageBox.Show("Cannot find the Model");
                        return;
                    }
                    Task.Run(() =>
                    {
                        Button_LoadModel.Invoke((Action)delegate
                        {
                            Button_LoadModel.Enabled = false;
                            TextBox_ModelPath.Enabled = false;
                            TextBox_VaePath.Enabled = false;
                            TextBox_LoraModelDir.Enabled = false;
                            CheckBox_CpuVae.Enabled = false;
                            CheckBox_VaeTiling.Enabled = false;
                            Button_ScanLoraPath.Enabled = false;
                            Button_ScanModelPath.Enabled = false;
                            Button_ScanVaePath.Enabled = false;
                        });
                        Structs.ModelParams modelParams = new Structs.ModelParams
                        {
                            ModelPath = modelPath,
                            VaePath = vaePath,
                            RngType = Structs.RngType.CUDA_RNG,
                            KeepVaeOnCpu = keepVaeOnCpu,
                            VaeTiling = vaeTiling,
                            LoraModelDir = loraModelDir,
                        };

                        bool result = helper.Initialize(modelParams);
                        Debug.WriteLine(result ? "Model loaded" : "Model not loaded");
                        Button_LoadModel.Invoke((Action)delegate
                        {
                            Button_LoadModel.Text = "Unload Model";
                            Button_LoadModel.Enabled = true;
                        });
                    });

                }
            }
            else
            {
                helper.FreeSD();
                Button_LoadModel.Text = "Load Model";
                TextBox_ModelPath.Enabled = true;
                TextBox_VaePath.Enabled = true;
                TextBox_LoraModelDir.Enabled = true;
                CheckBox_CpuVae.Enabled = true;
                CheckBox_VaeTiling.Enabled = true;
                Button_ScanLoraPath.Enabled = true;
                Button_ScanModelPath.Enabled = true;
                Button_ScanVaePath.Enabled = true;
            }
        }

        private void Helper_Progress(object sender, StableDiffusionEventArgs.StableDiffusionProgressEventArgs e)
        {
            base.Invoke((Action)delegate
              {
                  label11.Text = $"{e.Step}/{e.Steps}";
                  label14.Text = $"{e.IterationsPerSecond:f2} it/s";
                  ProgressBar_Progress.Value = (int)((e.Progress > 1 ? 1 : e.Progress) * 100);
              });
        }

        private void Button_TextToImage_Click(object sender, EventArgs e)
        {
            if (!helper.IsInitialized)
            {
                MessageBox.Show("Please load Model first");
                return;
            }

            Math.DivRem((int)NumericUpDown_Width.Value, 64, out int result1);
            Math.DivRem((int)NumericUpDown_Height.Value, 64, out int result2);
            if (result1 != 0 || result2 != 0)
            {
                MessageBox.Show("The width and height of the generated image must be a multiple of 64");
                return;
            }

            Button_TextToImage.Enabled = false;
            Button_ImageToImage.Enabled = false;

            Structs.TextToImageParams textToImageParams = new Structs.TextToImageParams
            {
                Prompt = TextBox_Prompt.Text,
                NegativePrompt = TextBox_NegativePrompt.Text,
                SampleMethod = (Structs.SampleMethod)Enum.Parse(typeof(Structs.SampleMethod), ComboBox_SampleMethod.Text),
                Width = (int)NumericUpDown_Width.Value,
                Height = (int)NumericUpDown_Height.Value,
                NormalizeInput = true,
                ClipSkip = (int)NumericUpDown_ClipSkip.Value,
                CfgScale = (float)NumericUpDown_CFG.Value,
                SampleSteps = (int)NumericUpDown_SampleSteps.Value,
                Seed = (long)NumericUpDown_Seed.Value,
            };
            Task.Run(() =>
            {
                Stopwatch stopwatch = new Stopwatch();
                stopwatch.Restart();
                Bitmap[] outputImages = helper.TextToImage(textToImageParams);
                for (int i = 0; i < outputImages.Length; i++)
                {
                    if (!Directory.Exists("output"))
                    {
                        Directory.CreateDirectory("output");
                    }
                    if (!Directory.Exists("./output/txt2img"))
                    {
                        Directory.CreateDirectory("./output/txt2img");
                    }
                    outputImages[i].Save($"./output/txt2img/{DateTime.Now:yyyyMMddHHmmss}-{i}.png");
                }
                base.Invoke((Action)delegate
                {
                    PictureBox_OutputImage.Image = outputImages[0];
                    Button_TextToImage.Enabled = true;
                    Button_ImageToImage.Enabled = true;
                });


                Debug.WriteLine($"Time to elapsed: {stopwatch.ElapsedMilliseconds} ms");
            });
        }

        private void MainForm_Load(object sender, EventArgs e)
        {
            SDHelper.Log += SDHelper_Log;
            SDHelper.Progress += Helper_Progress;

            ComboBox_SampleMethod.Items.AddRange(Enum.GetNames(typeof(Structs.SampleMethod)));
            ComboBox_SampleMethod.SelectedIndex = 0;

            PictureBox_InputImage.AllowDrop = true;
            PictureBox_UpscaleInput.AllowDrop = true;

        }

        private void SDHelper_Log(object sender, StableDiffusionEventArgs.StableDiffusionLogEventArgs e)
        {
            Console.WriteLine($"time:{DateTime.Now},  {e.Level}: {e.Text}");
            if (e.Text.Contains("vae compute"))
            {
                base.Invoke((Action)delegate { label12.Text = "VAE Progress"; });
            }
            else if (e.Text.Contains("generating image"))
            {
                base.Invoke((Action)delegate { label12.Text = "Generate Progress"; });
            }
        }

        private void Button_ScanModelPath_Click(object sender, EventArgs e)
        {
            FileDialog fileDialog = new OpenFileDialog
            {
                Filter = "Safetensors Files (*.safetensors)|*.safetensors|CheckPoint Files (*.ckpt)|*.ckpt|GGUF Files (*.gguf)|*.gguf|All Files (*.*)|*.*"
            };

            if (fileDialog.ShowDialog() == DialogResult.OK)
            {
                TextBox_ModelPath.Text = fileDialog.FileName;
            }
        }

        private void Button_ScanVaePath_Click(object sender, EventArgs e)
        {
            FileDialog fileDialog = new OpenFileDialog
            {
                Filter = "Safetensors Files (*.safetensors)|*.safetensors|CheckPoint Files (*.ckpt)|*.ckpt|GGUF Files (*.gguf)|*.gguf|All Files (*.*)|*.*"
            };
            if (fileDialog.ShowDialog() == DialogResult.OK)
            {
                TextBox_VaePath.Text = fileDialog.FileName;
            }
        }

        private void Button_ScanLoraPath_Click(object sender, EventArgs e)
        {
            FolderBrowserDialog folderBrowserDialog = new FolderBrowserDialog();
            if (folderBrowserDialog.ShowDialog() == DialogResult.OK)
            {
                TextBox_LoraModelDir.Text = folderBrowserDialog.SelectedPath;
            }
        }

        private void Button_ImageToImage_Click(object sender, EventArgs e)
        {
            if (!helper.IsInitialized)
            {
                MessageBox.Show("Please load a Model");
                return;
            }

            if (null == PictureBox_InputImage.Image)
            {
                MessageBox.Show("Please select an Image");
                return;
            }
            Bitmap inputBitmap = PictureBox_InputImage.Image.Clone() as Bitmap;
            Math.DivRem(inputBitmap.Width, 64, out int result1);
            Math.DivRem(inputBitmap.Height, 64, out int result2);
            if (result1 != 0 || result2 != 0)
            {
                MessageBox.Show("The width and height of the generated image must be a multiple of 64");
                return;
            }
            Button_TextToImage.Enabled = false;
            Button_ImageToImage.Enabled = false;
            Structs.ImageToImageParams imageToImageParams = new Structs.ImageToImageParams
            {
                InputImage = inputBitmap,
                Prompt = TextBox_Prompt.Text,
                NegativePrompt = TextBox_NegativePrompt.Text,
                CfgScale = (float)NumericUpDown_CFG.Value,
                Width = inputBitmap.Width,
                Height = inputBitmap.Height,
                SampleMethod = (Structs.SampleMethod)Enum.Parse(typeof(Structs.SampleMethod), ComboBox_SampleMethod.Text),
                SampleSteps = (int)NumericUpDown_SampleSteps.Value,
                Strength = (float)NumericUpDown_ReDrawStrength.Value,
                Seed = (long)NumericUpDown_Seed.Value,
                ClipSkip = (int)NumericUpDown_ClipSkip.Value,
            };
            Task.Run(() =>
            {
                Bitmap outputImage = helper.ImageToImage(imageToImageParams);

                if (!Directory.Exists("output"))
                {
                    Directory.CreateDirectory("output");
                }
                if (!Directory.Exists("./output/img2img"))
                {
                    Directory.CreateDirectory("./output/img2img");
                }
                outputImage.Save($"./output/img2img/{DateTime.Now:yyyyMMddHHmmss}.png");

                base.Invoke((Action)delegate
                {
                    PictureBox_OutputImage.Image = outputImage;
                    Button_TextToImage.Enabled = true;
                    Button_ImageToImage.Enabled = true;
                    Button_TextToImage.Enabled = true;
                });
            });

        }

        private void Form1_FormClosed(object sender, FormClosedEventArgs e)
        {
            helper.FreeSD();
            helper.FreeUpscaler();
            GC.Collect();
        }

        private void Button_RandomSeed_Click(object sender, EventArgs e)
        {
            Random random = new Random();
            int randomPositiveInteger = random.Next(1, int.MaxValue);
            NumericUpDown_Seed.Value = randomPositiveInteger;
        }

        private void PictureBox_InputImage_Click(object sender, EventArgs e)
        {
            OpenFileDialog openFileDialog = new OpenFileDialog
            {
                Filter = "Image Files (*.png, *.jpg, *.bmp)|*.png;*.jpg;*.bmp"
            };
            if (openFileDialog.ShowDialog() == DialogResult.OK)
            {
                Bitmap bitmap = new Bitmap(openFileDialog.FileName);
                PictureBox_InputImage.Image = bitmap;
            }
        }

        private void PictureBox_InputImage_DragEnter(object sender, DragEventArgs e)
        {
            if (e.Data.GetDataPresent(DataFormats.FileDrop))
                e.Effect = DragDropEffects.Link;
            else e.Effect = DragDropEffects.None;
        }

        private void PictureBox_InputImage_DragDrop(object sender, DragEventArgs e)
        {
            string fileName = ((Array)e.Data.GetData(DataFormats.FileDrop)).GetValue(0).ToString();
            Bitmap bitmap = new Bitmap(fileName);
            PictureBox_InputImage.Image = bitmap;
        }

        private void Button_ScanUpscaleModelPath_Click(object sender, EventArgs e)
        {
            OpenFileDialog openFileDialog = new OpenFileDialog
            {
                Filter = "ESRGAN Files (*.pth)|*.pth"
            };
            if (openFileDialog.ShowDialog() == DialogResult.OK)
            {
                TextBox_UpscaleModelPath.Text = openFileDialog.FileName;
            }
        }

        private void PictureBox_UpscaleInput_DragDrop(object sender, DragEventArgs e)
        {
            string fileName = ((Array)e.Data.GetData(DataFormats.FileDrop)).GetValue(0).ToString();
            Bitmap bitmap = new Bitmap(fileName);
            PictureBox_UpscaleInput.Image = bitmap;
        }

        private void PictureBox_UpscaleInput_DragEnter(object sender, DragEventArgs e)
        {
            if (e.Data.GetDataPresent(DataFormats.FileDrop))
                e.Effect = DragDropEffects.Link;
            else e.Effect = DragDropEffects.None;
        }

        private void Button_Upscale_Click(object sender, EventArgs e)
        {

            if (string.IsNullOrEmpty(TextBox_UpscaleModelPath.Text))
            {
                MessageBox.Show("Please select a upscale Model");
                return;
            }
            bool upscalerInited = helper.InitializeUpscaler(new Structs.UpscalerParams
            {
                ESRGANPath = TextBox_UpscaleModelPath.Text,

            });
            if (!upscalerInited)
            {
                MessageBox.Show("There is an error in loading upscale Model");
                return;
            }
            if (PictureBox_UpscaleInput.Image == null)
            {
                MessageBox.Show("Please select an Image");
                return;
            }
            Bitmap upscaleInputImage = PictureBox_UpscaleInput.Image as Bitmap;
            Button_Upscale.Enabled = false;
            Task.Run(() =>
            {
                try
                {
                    Button_Upscale.Enabled = false;
                    Bitmap bitmap = helper.UpscaleImage(PictureBox_UpscaleInput.Image as Bitmap, 4);
                    helper.FreeUpscaler();
                    if (!Directory.Exists("output"))
                    {
                        Directory.CreateDirectory("output");
                    }
                    if (!Directory.Exists("./output/upscale"))
                    {
                        Directory.CreateDirectory("./output/upscale");
                    }
                    bitmap.Save($"./output/upscale/{DateTime.Now:yyyyMMddHHmmss}.png");
                    base.Invoke((Action)delegate { PictureBox_UpscaleOutput.Image = bitmap; });
                }
                catch (Exception ex)
                {
                    MessageBox.Show(ex.Message);
                }
                finally
                {
                    base.Invoke((Action)delegate { Button_Upscale.Enabled = true; });

                    helper.FreeUpscaler();
                }

            });


        }

        private void PictureBox_UpscaleInput_Click(object sender, EventArgs e)
        {
            OpenFileDialog openFileDialog = new OpenFileDialog
            {
                Filter = "Image Files (*.png, *.jpg, *.bmp)|*.png;*.jpg;*.bmp"
            };
            if (openFileDialog.ShowDialog() == DialogResult.OK)
            {
                Bitmap bitmap = new Bitmap(openFileDialog.FileName);
                PictureBox_UpscaleInput.Image = bitmap;
            }
        }
    }
}
